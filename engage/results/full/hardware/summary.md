# Runtime on three devices (3 people in view, CPU, same code/models/frames/versions)

| Device | Networks (ms) | Ours, after the networks (ms) | Type-1 instead (ms) | fps | Ours share | Ours max fps |
|---|---|---|---|---|---|---|
| Raspberry Pi 5 (Raspberry Pi 5) | 474 | 1.49 | 1.30 | 2.1 | 0.31 % | 670 |
| Laptop CPU (AMD Ryzen 7 5825U with Radeon Graphics) | 161 | 0.95 | 0.83 | 6.2 | 0.59 % | 1055 |
| Desktop CPU (AMD Ryzen 9 9950X 16-Core Processor) | 64 | 0.35 | 0.29 | 15.5 | 0.54 % | 2870 |

GPU delegate: Laptop CPU (integrated graphics): failed; Desktop CPU (NVIDIA GeForce RTX 4080 SUPER): failed
