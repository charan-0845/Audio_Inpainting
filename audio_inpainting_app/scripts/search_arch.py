import sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.models.unet import PlainUNet
from src.models.multires_unet import MultiResUNet

TARGET_PLAIN = 2158578
TARGET_MULTI = 2015252

print("Evaluating closest architectures to paper...")

# PlainUNet
sched_plain = [68, 68, 68, 68, 68]
m_plain = PlainUNet(channel_schedule=sched_plain, norm="batch")
p_plain = m_plain.count_parameters()
print(f"PlainUNet {sched_plain}: {p_plain} params (Target: {TARGET_PLAIN}, Diff: {abs(p_plain - TARGET_PLAIN)})")

# MultiResUNet
sched_multi = [68, 68, 68, 68, 68]
m_multi = MultiResUNet(channel_schedule=sched_multi, res_path_lengths=[4,3,2,1], alpha=1.67, norm="batch")
p_multi = m_multi.count_parameters()
print(f"MultiResUNet {sched_multi}: {p_multi} params (Target: {TARGET_MULTI}, Diff: {abs(p_multi - TARGET_MULTI)})")

# Let's also check alpha=1.6 for MultiResUNet
m_multi2 = MultiResUNet(channel_schedule=sched_multi, res_path_lengths=[4,3,2,1], alpha=1.6, norm="batch")
p_multi2 = m_multi2.count_parameters()
print(f"MultiResUNet {sched_multi} alpha=1.6: {p_multi2} params (Target: {TARGET_MULTI}, Diff: {abs(p_multi2 - TARGET_MULTI)})")

