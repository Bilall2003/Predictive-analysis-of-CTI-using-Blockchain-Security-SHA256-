# CICIDS2017 Dataset — Class Distribution Overview

This document summarizes the exploratory data analysis (EDA) charts generated for the CICIDS2017 dataset used in the hybrid NIDS (Network Intrusion Detection System) FYP.

## 1. Label Distribution (Log Scale)

A bar chart showing the raw count of flows per traffic label, plotted on a log scale (y-axis: `10^1` to `10^6`) due to the extreme class imbalance in the dataset.

**Order (highest to lowest count):**
1. BENIGN
2. DoS Hulk
3. PortScan
4. DDoS
5. DoS GoldenEye
6. DoS slowloris
7. DoS Slowhttptest
8. FTP-Patator
9. SSH-Patator
10. Bot
11. Web Attack - Brute Force
12. Web Attack - XSS
13. Infiltration
14. Web Attack - Sql Injection
15. Heartbleed

BENIGN dominates by roughly an order of magnitude over the next-largest classes (DoS Hulk, PortScan), while rare attack types (Infiltration, SQL Injection, Heartbleed) have only single- or double-digit-thousands (or fewer) samples — motivating the use of SMOTE for minority-class oversampling.

## 2. BENIGN vs ATTACK (Pie Chart)

A simple binary split of the dataset:

| Class   | Percentage |
|---------|-----------|
| BENIGN  | 57.241%   |
| ATTACK  | 42.759%   |

## 3. Traffic Classes (Full Pie Chart with Counts)

A detailed breakdown of all 14 traffic classes with exact percentages and counts:

| Traffic Class            | Percentage | Count    |
|---------------------------|-----------|----------|
| BENIGN                     | 57.2412%  | 592,822  |
| DoS Hulk                   | 15.3572%  | 159,048  |
| PortScan                   | 15.3548%  | 159,023  |
| DDoS                        | 9.1848%   | 95,123   |
| DoS GoldenEye               | 0.7384%   | 7,647    |
| DoS slowloris                | 0.5511%   | 5,707    |
| DoS Slowhttptest            | 0.4933%   | 5,109    |
| FTP-Patator                 | 0.3847%   | 3,984    |
| SSH-Patator                 | 0.2885%   | 2,988    |
| Bot                         | 0.2132%   | 2,208    |
| Web Attack - Brute Force    | 0.1318%   | 1,365    |
| Web Attack - XSS            | 0.0542%   | 561      |
| Infiltration                | 0.0046%   | 48       |
| Web Attack - Sql Injection  | 0.0012%   | 12       |
| Heartbleed                  | 0.0011%   | 11       |

**Total flows: 1,036,646** (sum of all classes)

## 4. Median Flow Duration By Label (Log Scale)

A horizontal bar chart comparing the median flow duration (in microseconds, log scale) across labels, ordered from longest to shortest median duration:

**Order (longest to shortest):**
1. Heartbleed — ~10^8 (longest, near-persistent connections)
2. DoS slowloris — ~10^8
3. Infiltration — ~10^8
4. DoS Slowhttptest — ~10^8
5. SSH-Patator — ~10^7
6. DoS GoldenEye — ~10^7
7. FTP-Patator — ~10^7
8. DDoS — ~10^7
9. Web Attack - Brute Force — ~10^7
10. Web Attack - XSS — ~10^6
11. Web Attack - Sql Injection — ~10^6
12. DoS Hulk — ~10^5
13. BENIGN — ~10^5
14. Bot — ~10^2–10^3
15. PortScan — shortest (~10^1–10^2, near-instant scan probes)

**Key observation:** Slow-rate attacks (slowloris, Slowhttptest, Heartbleed, Infiltration) hold connections open deliberately, producing very long median flow durations. In contrast, PortScan and Bot traffic are characterized by short-lived flows, consistent with rapid probing/automated behavior. This duration signature is a useful feature for the NIDS classifier to distinguish attack types beyond simple packet/byte counts.

---
*Generated for the CICIDS2017 hybrid NIDS FYP dataset EDA.*
