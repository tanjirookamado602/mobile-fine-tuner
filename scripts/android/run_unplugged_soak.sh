#!/system/bin/sh
# Self-contained unplugged on-device RSS, CPU frequency, thermal & power benchmark.
# Runs entirely on the Android phone without USB connection.

WORKDIR="/data/local/tmp/mft_projector"
LOGDIR="$WORKDIR/unplugged_logs"
mkdir -p "$LOGDIR"
cd "$WORKDIR" || exit 1

echo "=== DEVICE HARDWARE & KERNEL INFO ===" > "$LOGDIR/device_hardware_info.txt"
date >> "$LOGDIR/device_hardware_info.txt"
echo "--- getprop ---" >> "$LOGDIR/device_hardware_info.txt"
for prop in ro.product.model ro.product.device ro.product.manufacturer ro.board.platform ro.hardware ro.mediatek.platform ro.soc.model ro.build.version.release; do
    echo "$prop = $(getprop $prop)" >> "$LOGDIR/device_hardware_info.txt"
done
echo "--- /proc/cpuinfo ---" >> "$LOGDIR/device_hardware_info.txt"
cat /proc/cpuinfo >> "$LOGDIR/device_hardware_info.txt"
echo "--- /proc/meminfo (before run) ---" >> "$LOGDIR/device_hardware_info.txt"
cat /proc/meminfo >> "$LOGDIR/device_hardware_info.txt"

TELEM_CSV="$LOGDIR/unplugged_telemetry.csv"
echo "phase,elapsed_s,status,capacity_pct,current_ua,voltage_uv,power_mw,temp_decidegc,cpu0_khz,cpu6_khz,vmrss_kb,vmhwm_kb" > "$TELEM_CSV"

sample_telemetry() {
    local phase="$1"
    local elapsed="$2"
    local pid="$3"
    local status=$(cat /sys/class/power_supply/battery/status 2>/dev/null | tr -d '\r\n')
    local cap=$(cat /sys/class/power_supply/battery/capacity 2>/dev/null | tr -d '\r\n')
    local curr=$(cat /sys/class/power_supply/battery/current_now 2>/dev/null | tr -d '\r\n')
    local volt=$(cat /sys/class/power_supply/battery/voltage_now 2>/dev/null | tr -d '\r\n')
    local temp=$(cat /sys/class/power_supply/battery/temp 2>/dev/null | tr -d '\r\n')
    local c0=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null | tr -d '\r\n')
    local c6=$(cat /sys/devices/system/cpu/cpu6/cpufreq/scaling_cur_freq 2>/dev/null | tr -d '\r\n')
    local rss="0"
    local hwm="0"
    if [ -n "$pid" ] && [ -d "/proc/$pid" ]; then
        rss=$(awk '/VmRSS:/ {print $2}' "/proc/$pid/status" 2>/dev/null)
        hwm=$(awk '/VmHWM:/ {print $2}' "/proc/$pid/status" 2>/dev/null)
    fi
    # power_mw = |curr_ua| * volt_uv / 1e9
    local pmw=$(awk -v c="$curr" -v v="$volt" 'BEGIN { if (c<0) c=-c; printf "%.2f", (c*v)/1000000000.0 }')
    echo "$phase,$elapsed,$status,$cap,$curr,$volt,$pmw,$temp,${c0:-0},${c6:-0},${rss:-0},${hwm:-0}" >> "$TELEM_CSV"
}

echo "[unplugged_soak] Waiting 20 seconds so you can UNPLUG the USB cable now..." > "$LOGDIR/soak_status.txt"
sleep 20

START_TS=$(date +%s)

# 1. Idle baseline for 60 seconds (unplugged)
i=0
while [ $i -lt 12 ]; do
    NOW_TS=$(date +%s)
    ELAPSED=$((NOW_TS - START_TS))
    sample_telemetry "idle_baseline" "$ELAPSED" ""
    sleep 5
    i=$((i + 1))
done

# 2. Launch train_projector for ~15 minutes (40 steps, val every 20)
LD_LIBRARY_PATH="$WORKDIR" "$WORKDIR/train_projector" \
    --load_quantized "$WORKDIR/models/qwen2.5-0.5b-q8_0.q8" \
    --model_dir "$WORKDIR/models/Qwen2.5-0.5B" \
    --train_path "$WORKDIR/data/multimodal_splits/train.jsonl" \
    --val_path "$WORKDIR/data/multimodal_splits/val.jsonl" \
    --features_path "$WORKDIR/data/vision_features/visual_features.npy" \
    --steps 40 \
    --val_every 20 \
    --lr 0.001 > "$LOGDIR/train_unplugged.log" 2>&1 &
TPID=$!

echo "[unplugged_soak] Started train_projector PID=$TPID" >> "$LOGDIR/soak_status.txt"

# Monitor up to 900s (15 min) or until training finishes
while kill -0 "$TPID" 2>/dev/null; do
    NOW_TS=$(date +%s)
    ELAPSED=$((NOW_TS - START_TS))
    sample_telemetry "training" "$ELAPSED" "$TPID"
    if [ "$ELAPSED" -ge 960 ]; then
        echo "[unplugged_soak] Reached 15m training window, stopping PID=$TPID cleanly..." >> "$LOGDIR/soak_status.txt"
        kill "$TPID" 2>/dev/null
        break
    fi
    sleep 5
done

wait "$TPID" 2>/dev/null
NOW_TS=$(date +%s)
ELAPSED=$((NOW_TS - START_TS))
sample_telemetry "post_finish" "$ELAPSED" ""
echo "[unplugged_soak] Complete at elapsed=${ELAPSED}s" >> "$LOGDIR/soak_status.txt"
cat /proc/meminfo > "$LOGDIR/meminfo_after.txt"
