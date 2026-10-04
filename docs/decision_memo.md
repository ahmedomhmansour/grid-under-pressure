# Decision memo: Meeting Dominion-zone load growth through 2030

**To:** Resource planning leadership (illustrative audience)
**From:** Ahmed Mansour
**Date:** October 2026

## Recommendation
Plan for about **5.6 GW of new accredited capacity by 2030**, not the 3.5 GW that PJM's standard planning requirement implies. Alternatively, pair the 3.5 GW build with a **winterization program** for the existing gas fleet. Under the model's assumptions, the 3.5 GW plan alone leaves the zone about five times short of the 1-day-in-10-years reliability standard.

## Situation
- Dominion's winter peak load rose from 17.5 GW (2019–20) to 25.4 GW (2025–26), up 45% (EIA-930). Most of the growth is data centers.
- PJM's 2026 forecast implies about 3.9%/yr (summer) to 4.7%/yr (winter) peak growth for the zone; Dominion's own 2025 IRP Update says 2.5%/yr for its service territory. I test 2.5% / 4.7% / 6.5%.
- The zone cleared 21,246 MW of accredited capacity in PJM's 2028/29 auction and can import up to 6,441 MW in an emergency.

## Findings
1. **Scale:** at 4.7%/yr, the weather-normal peak reaches 32.4 GW by 2030, 6.6 GW above 2025.
2. **Least-cost mix:** meeting PJM's requirement (accredited capacity ≥ 0.94 × peak) takes 3.0 GW of gas combined cycle plus 0.4 GW of demand response, for a total system cost of $5.78B/yr.
3. **Reliability gap:** across 1,000 simulated years, that plan loses load 0.51 days/yr against a 0.1 target. Almost all of the shortfall happens in cold snaps, when demand peaks and gas plants fail together.
4. **Cost of closing it:** raising the requirement to 1.00 × peak (5.6 GW of new build) meets the standard for $125M/yr more, a 2.2% increase in system cost.
5. **Solar and storage:** without tax credits, neither is part of the least-cost plan. Solar costs about $85/MWh against $31/MWh to run existing gas, and PJM credits it with only 10% of its capacity at peak.

## Risks and what would change the answer
- **Cold-weather performance is the swing factor.** Without correlated cold-weather outages, the 3.5 GW plan passes (0.09 days/yr). Halving the effect still fails (0.25 days/yr).
- **Load growth** has been revised repeatedly. At Dominion's 2.5%/yr, no new build is needed to meet PJM's requirement, but the reliability gap remains.
- **Imports** during region-wide stress are uncertain. If the full import limit is always deliverable, LOLE falls to 0.32 days/yr, still above the standard.

## Method
A gradient-boosted load-shape model was trained on 2019–2025 EIA-930 hourly load and ERA5 weather. Its out-of-sample error was 3.0–3.2%, against 6.3–6.4% for a linear degree-day benchmark. A least-cost capacity-expansion linear program (PuLP/HiGHS) was solved on 16 representative days, with PJM's 2028/29 ELCC ratings and reliability requirement. A Monte Carlo resource-adequacy check over 1,000 simulated years covered weather-year, load, unit-outage and cold-weather uncertainty. All assumptions and their sources are in `config/assumptions.yaml`.
