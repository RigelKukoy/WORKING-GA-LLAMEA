"""
Generate the Section 4.5 Behaviour Profile Analysis FINAL Report (PDF).
All numerical values verified from extract_data_for_section.ipynb outputs.
Run: python generate_section45_report.py
"""
from fpdf import FPDF
import os

OUT_PATH = "Section_4.5_Behaviour_Profile_Analysis_FINAL.pdf"

# ── PDF helper ──────────────────────────────────────────────────────────────────
class Report(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 6, "GA-LLaMEA  |  Section 4.5 Behaviour Profile Analysis  |  FINAL", align="C")
        self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title, level=1):
        sizes = {1: 14, 2: 12, 3: 11}
        self.set_font("Helvetica", "B", sizes.get(level, 11))
        self.set_text_color(20, 60, 120)
        self.ln(4)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(20, 60, 120)
        if level == 1:
            self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(3)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5.2, text)
        self.ln(2)

    def italic_text(self, text):
        self.set_font("Helvetica", "I", 10)
        self.set_text_color(60, 60, 60)
        self.multi_cell(0, 5.2, text)
        self.ln(2)

    def bold_text(self, text):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5.2, text)
        self.ln(1)

    def add_table(self, headers, rows, col_widths=None):
        if col_widths is None:
            usable = self.w - self.l_margin - self.r_margin
            col_widths = [usable / len(headers)] * len(headers)
        # Header
        self.set_font("Helvetica", "B", 8)
        self.set_fill_color(20, 60, 120)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, str(h), border=1, fill=True, align="C")
        self.ln()
        # Rows
        self.set_font("Helvetica", "", 8)
        self.set_text_color(30, 30, 30)
        for ri, row in enumerate(rows):
            fill = ri % 2 == 0
            if fill:
                self.set_fill_color(235, 240, 250)
            for i, val in enumerate(row):
                align = "L" if i == 0 else "C"
                self.cell(col_widths[i], 6, str(val), border=1, fill=fill, align=align)
            self.ln()
        self.ln(3)


# ── Build report ────────────────────────────────────────────────────────────────
pdf = Report()
pdf.alias_nb_pages()
pdf.set_auto_page_break(auto=True, margin=20)
pdf.add_page()

# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT A  --  Analysis Summary
# ═══════════════════════════════════════════════════════════════════════════════
pdf.section_title("Output A -- Analysis Summary", 1)

pdf.section_title("A.1  Methodology Adopted from van Stein et al. (2025)", 2)
pdf.body_text(
    "The behaviour-space analysis follows the two-step methodology introduced by van Stein et al. (2025) "
    "in \"Behaviour Space Analysis of LLM-driven Meta-heuristic Discovery.\" That study defines 11 scalar "
    "behavioural metrics computed from the optimisation trace of each generated algorithm, grouped into four "
    "categories: Exploration & Diversity (NN-dist, Disp, Expl%), Exploitation & Intensification (Dist->best, "
    "Inten-ratio, Explt%), Convergence Progress (Conv-rate, Delta fitness, Success%), and Stagnation & "
    "Reliability (No-imp streak, Last-imp frac)."
)
pdf.body_text(
    "Step 1 -- Correlation Heatmap (Feature Selection): Pearson r is computed between all 11 metrics plus "
    "normalised AOCC fitness. Redundant pairs (|r| >= 0.7) are identified and one feature from each pair is "
    "dropped, retaining the one more strongly correlated with fitness. This yields a compact, non-redundant "
    "feature set for profiling."
)
pdf.body_text(
    "Step 2 -- Parallel Coordinates (Gold Standard Profile): All generated algorithms are plotted as "
    "polylines across the selected metrics, coloured by AOCC quartile. The Q4 (top 25%) cluster defines "
    "the 'gold standard' behaviour profile -- the set of metric values associated with high-performing "
    "algorithms. Each method's median profile is then compared against this gold standard to diagnose "
    "behavioural strengths and weaknesses."
)

pdf.section_title("A.2  Adaptation to the GA-LLaMEA Experiment", 2)
pdf.body_text(
    "We apply this methodology to three methods: Evolution of Heuristics (EoH), Baseline LLaMEA, and "
    "GA-LLaMEA (with 8-parent initialisation and 100 evaluations). Each method generated approximately "
    "500 algorithms over 5 independent runs on MA-BBOB (5D). Every generated algorithm was re-executed on "
    "5 MA-BBOB instances x 2 repetitions using IOH Experimenter, and the 11 behaviour metrics were computed "
    "from the resulting traces. Metrics were aggregated (median) over all runs per algorithm, yielding one "
    "behaviour vector per algorithm -- 1,442 algorithms in total (472 Baseline-LLaMEA, 482 EoH, 488 GA-LLaMEA)."
)

pdf.section_title("A.3  Notebook Outputs Used", 2)
pdf.body_text(
    "The analysis draws on the following notebook outputs: (1) Behaviour DataFrame with 1,442 algorithms x 15 "
    "columns; (2) Correlation heatmap of all 11 metrics + normalised AOCC; (3) Redundant feature pair table "
    "(20 pairs with |r| >= 0.7); (4) Parallel coordinate plots -- all algorithms by quartile, top-100 per "
    "method, and per-method quartile profiles; (5) Gold standard comparison table (median metrics per method "
    "vs. Q4 gold standard); (6) 5 non-redundant feature profile plot; (7) Mean +/- Std breakdown per method "
    "for all 11 features; (8) Per-method Pearson correlations with normalised AOCC; (9) Q4 algorithm counts "
    "and proportions per method."
)

# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT B  --  Proposed Section Naming
# ═══════════════════════════════════════════════════════════════════════════════
pdf.add_page()
pdf.section_title("Output B -- Proposed Section Naming", 1)

pdf.bold_text("Recommended Section Title:")
pdf.body_text("4.5  Behaviour Profile Analysis")

pdf.bold_text("Recommended Subheadings:")
pdf.body_text(
    "4.5.1  Metric Correlation and Feature Selection\n"
    "4.5.2  Gold Standard Behaviour Profile\n"
    "4.5.3  Comparative Behaviour Profiles of EoH, LLaMEA, and GA-LLaMEA\n"
    "4.5.4  Behavioural Interpretation of Performance Differences"
)

pdf.bold_text("Justification:")
pdf.body_text(
    "The title \"Behaviour Profile Analysis\" directly mirrors the terminology used throughout the manuscript "
    "(Sections 2.9, 3.6) and aligns with the reference paper's framing. The subheadings follow the two-step "
    "methodology (correlation analysis then profiling) before moving to per-method comparison and interpretation. "
    "This structure parallels Section 4.3 of van Stein et al. (2025) and fits naturally between the existing "
    "Section 4.4 (Convergence and Reliability Behaviour) and Section 4.6 (CEG Behaviour) in the manuscript, "
    "since it transitions from per-algorithm convergence metrics to population-level behavioural profiling."
)

# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT C  --  Essential Figures and Tables
# ═══════════════════════════════════════════════════════════════════════════════
pdf.add_page()
pdf.section_title("Output C -- Essential Figures and Tables", 1)

pdf.section_title("C.1  Required Figures", 2)

fig_headers = ["#", "Title", "Shows", "Ref. in 4.5", "Status"]
fig_widths = [8, 55, 55, 30, 22]
fig_rows = [
    ["F10", "Correlation Heatmap of Behaviour Metrics",
     "Pearson r among 11 metrics + AOCC; identifies redundant pairs",
     "Sec 4.5.1", "Required"],
    ["F11", "Behaviour Profile: All Algorithms by AOCC Quartile",
     "Parallel coords coloured by Q1-Q4; reveals gold standard profile",
     "Sec 4.5.2", "Required"],
    ["F12", "Top-100 Behaviour Profiles per Method",
     "Parallel coords of best 100 algos per method; direct method comparison",
     "Sec 4.5.3", "Required"],
    ["F13", "Behaviour Profile: 5 Non-Redundant STN Features",
     "Reduced-dimension parallel coords confirming gold standard on key axes",
     "Sec 4.5.2", "Optional"],
]
pdf.add_table(fig_headers, fig_rows, fig_widths)

pdf.section_title("C.2  Required Tables", 2)

tbl_headers = ["#", "Title", "Shows", "Ref. in 4.5", "Status"]
tbl_widths = [8, 55, 55, 30, 22]
tbl_rows = [
    ["T5", "Median Behaviour Metrics per Method vs. Q4 Gold Standard",
     "Quantifies deviation of each method from the ideal behavioural profile",
     "Sec 4.5.3", "Required"],
    ["T6", "Mean and Std of Behaviour Metrics per Method",
     "Full distributional summary; supports variance-based claims",
     "Sec 4.5.3", "Required"],
    ["T7", "Redundant Feature Pairs (|r| >= 0.7)",
     "Documents feature selection rationale; 20 pairs identified",
     "Sec 4.5.1", "Optional"],
]
pdf.add_table(tbl_headers, tbl_rows, tbl_widths)

pdf.section_title("C.3  Optional / Supplementary Visuals", 2)
fig_opt_rows = [
    ["S1", "Per-Method Parallel Coordinate Profiles (3 panels)",
     "Within-method Q1-Q4 variation; diagnoses internal diversity",
     "Sec 4.5.3", "Supplem."],
]
pdf.add_table(fig_headers, fig_opt_rows, fig_widths)

# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT D  --  Copy-Paste-Ready Manuscript Text
# ═══════════════════════════════════════════════════════════════════════════════
pdf.add_page()
pdf.section_title("Output D -- Copy-Paste-Ready Manuscript Text for Section 4.5", 1)
pdf.italic_text(
    "[All numerical values in this section have been verified against the confirmed notebook outputs "
    "from extract_data_for_section.ipynb and the behaviour_profile_and_correlation.ipynb experiment.]"
)
pdf.ln(2)

# ── 4.5 ──
pdf.section_title("4.5  Behaviour Profile Analysis", 2)
pdf.body_text(
    "To move beyond aggregate performance metrics and understand how each method searches, "
    "we conduct a behaviour-space analysis following the methodology of van Stein et al. (2025). "
    "Every algorithm generated during evolution -- not only the final best -- is characterised by "
    "a vector of 11 behavioural metrics computed from its optimisation trace on MA-BBOB instances. "
    "These metrics span four categories: exploration and diversity (Nearest-Neighbour Distance, "
    "Coverage Dispersion, Exploration Percentage), exploitation and intensification (Distance-to-Best, "
    "Intensification Ratio, Exploitation Percentage), convergence progress (Convergence Rate, "
    "Average Improvement, Success Rate), and stagnation and reliability (Longest No-Improvement "
    "Streak, Last Improvement Fraction). In total, 1,442 algorithms were profiled: 472 from "
    "Baseline LLaMEA, 482 from EoH, and 488 from GA-LLaMEA."
)

# ── 4.5.1 ──
pdf.section_title("4.5.1  Metric Correlation and Feature Selection", 3)
pdf.body_text(
    "A Pearson correlation analysis was first performed across all 11 behaviour metrics and "
    "the normalised AOCC fitness score to verify metric complementarity and identify redundancies "
    "(Figure 10). As expected, Exploration Percentage and Exploitation Percentage are perfectly "
    "inversely correlated (r = -1.00) by construction. Several other strong correlations were "
    "observed: NN-dist correlates highly with Exploration Percentage (r = 0.88) and with "
    "Convergence Rate (r = -0.96); Distance-to-Best mirrors Exploration Percentage (r = 0.98); "
    "and Intensification Ratio is strongly negatively correlated with Exploration Percentage "
    "(r = -0.92). In total, 20 feature pairs exceeded the redundancy threshold of |r| >= 0.7."
)
pdf.body_text(
    "Among the correlations with normalised fitness, the strongest positive association was "
    "observed for Intensification Ratio (r = 0.551) and Exploitation Percentage (r = 0.460), while "
    "the strongest negative associations were found for Exploration Percentage (r = -0.460) and "
    "Distance-to-Best (r = -0.452). These correlations indicate that exploitation-focused search "
    "dynamics are strongly associated with higher AOCC performance across all three methods "
    "(van Stein et al., 2025). "
    "Notably, the per-method analysis reveals that these correlations are most pronounced for "
    "GA-LLaMEA (e.g., Intensification Ratio: r = 0.667; Exploration Percentage: r = -0.622) "
    "compared to Baseline LLaMEA (r = 0.453; r = -0.372) and EoH (r = 0.509; r = -0.351), "
    "suggesting that GA-LLaMEA's search process is more tightly governed by exploitation-oriented "
    "dynamics."
)
pdf.body_text(
    "Following the feature-selection protocol of van Stein et al. (2025), we retain from each "
    "redundant pair the feature with the stronger absolute correlation to fitness. The resulting "
    "five non-redundant features used for subsequent profiling are: Exploration Percentage, "
    "Convergence Rate, Average Improvement (Delta fitness), Success Rate, and Longest "
    "No-Improvement Streak -- identical to those selected in the reference study (van Stein et al., 2025)."
)

# ── 4.5.2 ──
pdf.section_title("4.5.2  Gold Standard Behaviour Profile", 3)
pdf.body_text(
    "Figure 11 presents parallel coordinate plots of all 1,442 algorithms coloured by normalised "
    "AOCC quartile (Q4 threshold: 0.8802). The top-performing quartile (Q4, dark red) reveals a "
    "clear gold standard profile: low NN-dist (median 0.1018), moderate-to-high Dispersion "
    "(median 7.4230), low Exploration Percentage (median 4.03%), high Intensification Ratio "
    "(median 0.9395), high Exploitation Percentage (median 95.97%), near-maximal Convergence Rate "
    "(0.9962), low Delta Fitness (0.1122), moderate Success Rate (0.92%), high No-Improvement "
    "Streak (4,482), and high Last Improvement Fraction (0.5252). This profile characterises "
    "algorithms that conduct focused, exploitation-heavy search near promising regions while "
    "sustaining slow but persistent improvement over extended periods (van Stein et al., 2025)."
)
pdf.body_text(
    "The gold standard profile observed in our experiment is consistent with the findings of "
    "van Stein et al. (2025), who reported that high-performing LLM-generated algorithms exhibit "
    "low nearest-neighbour distance, average dispersion, low exploration percentage, and a focused "
    "exploitation strategy. The high No-Improvement Streak in Q4 algorithms may appear "
    "counterintuitive but reflects the behaviour of well-converged algorithms that have already "
    "reached near-optimal solutions and make only marginal improvements in the later stages "
    "of the search budget."
)

# ── 4.5.3 ──
pdf.add_page()
pdf.section_title("4.5.3  Comparative Behaviour Profiles of EoH, LLaMEA, and GA-LLaMEA", 3)
pdf.body_text(
    "Table 5 presents the median behaviour metrics for each method alongside the Q4 gold standard. "
    "GA-LLaMEA achieves the closest alignment with the gold standard profile across the majority "
    "of metrics. Specifically, GA-LLaMEA records the lowest median Exploration Percentage (6.66%) "
    "among the three methods, compared to 9.70% for Baseline LLaMEA and 10.95% for EoH. "
    "Correspondingly, GA-LLaMEA exhibits the highest Exploitation Percentage (93.34%), the highest "
    "Intensification Ratio (0.8927), and the lowest Distance-to-Best (0.4241), indicating that "
    "its generated algorithms concentrate search effort near the best-found solutions more "
    "effectively than the baselines."
)

# Gold standard table -- VALUES VERIFIED from extract_data_for_section.ipynb TABLE 1
pdf.bold_text("Table 5. Median Behaviour Metrics per Method vs. Q4 Gold Standard")
headers = ["Metric", "Baseline-LLaMEA", "EoH", "GA-LLaMEA", "Q4 Gold Std"]
widths = [30, 35, 30, 30, 30]
rows = [
    ["NN-dist",        "0.1827", "0.1937", "0.1358", "0.1018"],
    ["Disp",           "6.9467", "7.0195", "6.9970", "7.4230"],
    ["Expl %",         "9.7038", "10.9453", "6.6566", "4.0268"],
    ["Dist->best",     "0.6312", "0.7277", "0.4241", "0.2441"],
    ["Inten-ratio",    "0.8362", "0.7761", "0.8927", "0.9395"],
    ["Explt %",        "90.2962", "89.0547", "93.3434", "95.9732"],
    ["Conv-rate",      "0.9962", "0.9962", "0.9962", "0.9962"],
    ["Delta fitness",  "0.1681", "0.1437", "0.1242", "0.1122"],
    ["Success %",      "0.0089", "0.0085", "0.0091", "0.0092"],
    ["No-imp streak",  "1753.25", "1924.25", "2795.00", "4482.00"],
    ["Last-imp frac",  "0.2677", "0.1436", "0.3501", "0.5252"],
]
pdf.add_table(headers, rows, widths)

# Mean +/- Std table -- VALUES from extract_data_for_section.ipynb TABLE 2
pdf.bold_text("Table 6. Mean +/- Standard Deviation of Behaviour Metrics per Method")
headers2 = ["Metric", "Baseline-LLaMEA", "EoH", "GA-LLaMEA"]
widths2 = [28, 44, 44, 44]
rows2 = [
    ["NN-dist",       "1.2481 +/- 2.0364", "0.6826 +/- 1.3052", "0.8142 +/- 1.6164"],
    ["Disp",          "7.0627 +/- 1.2045", "6.9334 +/- 1.8305", "6.8428 +/- 1.2758"],
    ["Expl %",        "31.19 +/- 37.25",   "26.25 +/- 31.19",   "22.79 +/- 32.99"],
    ["Dist->best",    "2.4177 +/- 3.1307", "2.0958 +/- 2.7642", "1.7714 +/- 2.8260"],
    ["Inten-ratio",   "0.6095 +/- 0.3895", "0.6093 +/- 0.3684", "0.7163 +/- 0.3502"],
    ["Explt %",       "68.81 +/- 37.25",   "73.75 +/- 31.19",   "77.21 +/- 32.99"],
    ["Conv-rate",     "0.8513 +/- 0.3007", "0.9476 +/- 0.1657", "0.9189 +/- 0.2172"],
    ["Delta fit.",    "1.0372 +/- 1.6494", "0.5640 +/- 1.0953", "0.6556 +/- 1.3208"],
    ["Success %",     "0.0238 +/- 0.0338", "0.0241 +/- 0.0638", "0.0169 +/- 0.0223"],
    ["No-imp str.",   "2538 +/- 2351",     "2655 +/- 2288",     "2976 +/- 2303"],
    ["Last-imp fr.",  "0.3127 +/- 0.2601", "0.2550 +/- 0.2663", "0.3342 +/- 0.2360"],
]
pdf.add_table(headers2, rows2, widths2)

pdf.body_text(
    "GA-LLaMEA also achieves the highest median AOCC (0.7376), substantially exceeding "
    "Baseline LLaMEA (0.5646) and EoH (0.4936). The mean AOCC values confirm this ordering: "
    "GA-LLaMEA (0.6028) > Baseline LLaMEA (0.5315) > EoH (0.5081). This performance advantage "
    "is directly reflected in its behavioural profile: GA-LLaMEA algorithms are more "
    "exploitation-focused, maintain tighter search around promising solutions, and sustain "
    "improvement for a larger fraction of the evaluation budget."
)
pdf.body_text(
    "The proportion of algorithms reaching Q4 performance further quantifies this advantage. "
    "GA-LLaMEA places 177 of its 488 algorithms (36.3%) in the top quartile, compared to 99 of "
    "472 (21.0%) for Baseline LLaMEA and only 85 of 482 (17.6%) for EoH. This indicates that "
    "GA-LLaMEA not only produces the best individual algorithms but also generates a substantially "
    "higher proportion of high-performing candidates throughout its evolutionary search."
)
pdf.body_text(
    "Figure 12 reinforces this finding by comparing the top-100 algorithms per method. "
    "GA-LLaMEA's top algorithms (green) form a tight, coherent cluster at low NN-dist, low "
    "Exploration Percentage, and high Intensification Ratio, closely matching the Q4 gold "
    "standard. In contrast, EoH's top algorithms (orange) display substantially greater spread "
    "across exploration-related axes, and Baseline LLaMEA (blue) occupies an intermediate "
    "position. This visual evidence confirms that GA-LLaMEA produces more behaviourally "
    "consistent high-performing algorithms."
)

pdf.body_text(
    "EoH exhibits the most exploratory profile among the three methods. Its median Exploration "
    "Percentage (10.95%) is the highest, and its Intensification Ratio (0.7761) is the lowest. "
    "EoH also records the highest Distance-to-Best (0.7277) and the lowest Last Improvement "
    "Fraction (0.1436), indicating that its algorithms make their final improvements relatively "
    "early in the search budget and spend a larger proportion of evaluations on broad exploration. "
    "This is consistent with EoH's design as a dual-population framework that jointly evolves "
    "both natural-language thoughts and code (Liu et al., 2024), promoting diversity but potentially at the cost "
    "of sustained exploitation."
)
pdf.body_text(
    "Baseline LLaMEA falls between the two on most metrics. Its Exploration Percentage (9.70%) "
    "is lower than EoH's but higher than GA-LLaMEA's, and its Intensification Ratio (0.8362) "
    "similarly occupies the middle ground. However, Baseline LLaMEA records the lowest "
    "No-Improvement Streak (1,753), suggesting that its mutation-only search (van Stein & Back, 2025) recovers from "
    "stagnation more quickly -- though this also implies it has not converged as deeply, as "
    "reflected in its lower median AOCC."
)

# ── 4.5.4 ──
pdf.section_title("4.5.4  Behavioural Interpretation of Performance Differences", 3)
pdf.body_text(
    "The behaviour profile analysis provides a mechanistic explanation for the performance "
    "hierarchy observed in earlier sections. GA-LLaMEA's superior AOCC performance can be "
    "attributed to its closer adherence to the exploitation-focused gold standard profile. "
    "Three behavioural factors appear to drive this advantage."
)
pdf.body_text(
    "First, the generative crossover mechanism enables GA-LLaMEA to combine complementary "
    "algorithmic strategies from multiple parents (Sudholt, 2017), producing offspring that inherit effective "
    "exploitation behaviours. This is reflected in the consistently high Intensification Ratio "
    "(median 0.8927, mean 0.7163) and low Distance-to-Best (median 0.4241, mean 1.7714). "
    "Unlike Baseline LLaMEA, whose mutation-only refinement is constrained to local perturbations "
    "of a single parent, generative crossover permits larger, semantically meaningful transitions "
    "in algorithm space that nonetheless preserve exploitation-oriented search dynamics. The "
    "per-method correlation data corroborates this: GA-LLaMEA exhibits the strongest coupling "
    "between Intensification Ratio and fitness (r = 0.667), indicating that its exploitation "
    "behaviour translates more reliably into performance gains."
)
pdf.body_text(
    "Second, the Discounted Thompson Sampling controller in GA-LLaMEA (Sun & Li, 2020; Tian et al., 2022) "
    "adaptively balances exploration and exploitation at the operator level. As demonstrated in Section 4.6, the "
    "controller progressively shifts selection probability toward the Simplify and Crossover "
    "operators, which produce algorithms with stronger exploitation profiles. This adaptive "
    "scheduling effectively steers the meta-level search toward the gold standard region of "
    "behaviour space."
)
pdf.body_text(
    "Third, GA-LLaMEA's algorithms sustain improvement later into the evaluation budget, as "
    "evidenced by the highest median Last Improvement Fraction (0.3501 vs. 0.2677 for LLaMEA "
    "and 0.1436 for EoH). This indicates that GA-LLaMEA produces algorithms with more effective "
    "late-stage refinement capability, consistent with the observation that exploitation-heavy "
    "algorithms continue to make incremental progress near the optimum rather than stagnating "
    "prematurely."
)
pdf.body_text(
    "Conversely, EoH's relatively poor median performance is explained by its overly exploratory "
    "profile. While exploration is essential in the early stages of optimisation, the behaviour "
    "metrics reveal that EoH algorithms do not transition as effectively to exploitation. This "
    "mirrors the diagnostic pattern identified by van Stein et al. (2025) for LLaMEA-2, whose "
    "pure random-new strategy led to excessive exploration, low success rate, and high stagnation. "
    "The per-method parallel coordinate plots further confirm that EoH's Q4 algorithms show wider "
    "behavioural variance than GA-LLaMEA's Q4 cluster, suggesting lower reliability in producing "
    "consistently exploitation-focused heuristics. Furthermore, only 17.6% of EoH's algorithms "
    "reach Q4 performance, compared to 36.3% for GA-LLaMEA, quantifying this reliability gap."
)
pdf.body_text(
    "These findings corroborate the theoretical motivation underpinning GA-LLaMEA: generative "
    "crossover combined with adaptive operator selection produces algorithms that more closely "
    "match the behavioural profile associated with high optimisation performance, thereby "
    "explaining not only that GA-LLaMEA outperforms the baselines, but why it does so."
)

# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT E  --  Figure/Table Placement Guide
# ═══════════════════════════════════════════════════════════════════════════════
pdf.add_page()
pdf.section_title("Output E -- Figure and Table Placement Guide", 1)

pdf.bold_text("Recommended Figure Numbers (continuing from existing manuscript):")
pdf.body_text(
    "Figure 10 -- Behaviour Metrics Correlation Heatmap\n"
    "  Caption: Pearson r correlation between the 11 behaviour metrics and normalised AOCC fitness. "
    "Strong redundancies (|r| >= 0.7) are visible between exploration- and exploitation-related metrics.\n\n"
    "Figure 11 -- Behaviour Profile of All Generated Algorithms\n"
    "  Caption: Parallel coordinate plot of all 1,442 generated algorithms across 11 behaviour metrics, "
    "coloured by normalised AOCC quartile (Q1 blue = low, Q4 dark red = high). The Q4 cluster defines "
    "the gold standard behaviour profile.\n\n"
    "Figure 12 -- Top-100 Behaviour Profiles per Method\n"
    "  Caption: Parallel coordinate plot of the 100 highest-AOCC algorithms per method. GA-LLaMEA (green) "
    "forms the tightest cluster, closely matching the Q4 gold standard profile.\n\n"
    "Figure 13 (Optional) -- Behaviour Profile on 5 Non-Redundant STN Features\n"
    "  Caption: Parallel coordinate plot using the 5 least-correlated behaviour metrics "
    "(Expl%, Conv-rate, Delta fitness, Success%, No-imp streak) coloured by AOCC quartile."
)

pdf.bold_text("Recommended Tables:")
pdf.body_text(
    "Table 5 -- Median Behaviour Metrics per Method vs. Q4 Gold Standard\n"
    "  Caption: Median values of all 11 behaviour metrics for each method compared to the Q4 (top 25%) "
    "gold standard. GA-LLaMEA achieves the closest alignment on 9 of 11 metrics.\n\n"
    "Table 6 -- Mean and Standard Deviation of Behaviour Metrics per Method\n"
    "  Caption: Mean +/- standard deviation of all 11 behaviour metrics across all generated algorithms "
    "per method, providing a full distributional summary of search behaviour."
)

pdf.ln(4)
pdf.section_title("Verified Data Sources", 2)
pdf.body_text(
    "All numerical values in this report have been cross-verified against two independent notebook runs:\n\n"
    "1. behaviour_profile_and_correlation.ipynb -- Primary experiment notebook (1,442 algorithms profiled "
    "from 1,497 generated; 472 Baseline-LLaMEA, 482 EoH, 488 GA-LLaMEA).\n\n"
    "2. extract_data_for_section.ipynb -- Verification notebook producing Tables 1-6 with confirmed "
    "values for all medians, means, standard deviations, correlations, and Q4 counts.\n\n"
    "Key confirmed statistics:\n"
    "  - Q4 normalised AOCC threshold: 0.8802\n"
    "  - GA-LLaMEA Q4 proportion: 177/488 (36.3%)\n"
    "  - Baseline-LLaMEA Q4 proportion: 99/472 (21.0%)\n"
    "  - EoH Q4 proportion: 85/482 (17.6%)\n"
    "  - Median AOCC: GA-LLaMEA 0.7376, Baseline-LLaMEA 0.5646, EoH 0.4936\n"
    "  - Mean AOCC: GA-LLaMEA 0.6028, Baseline-LLaMEA 0.5315, EoH 0.5081\n"
    "  - Max AOCC: GA-LLaMEA 0.8640, Baseline-LLaMEA 0.8566, EoH 0.8373"
)

pdf.section_title("Limitations and Caveats", 2)
pdf.body_text(
    "1. GA-LLaMEA has 488 algorithms with trajectory data out of 497 generated, due to 9 algorithms "
    "that crashed before completing any IOH evaluation runs. EoH and Baseline LLaMEA have slightly "
    "fewer (482 and 472 out of 500 respectively) for similar reasons. This represents >94% coverage "
    "for all methods.\n\n"
    "2. Trajectory data was collected on 5 MA-BBOB instances x 2 repetitions per algorithm (10 runs). "
    "The reference paper used 5 instances x 5 BBOB functions = 25 runs. Our smaller sample may "
    "introduce slightly higher variance in per-algorithm metric estimates.\n\n"
    "3. The correlation and feature selection were performed on the pooled dataset across all three "
    "methods. Per-method correlations differ in magnitude (GA-LLaMEA shows stronger correlations "
    "overall) but the directional patterns are consistent across all methods."
)

# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT F  --  Supporting References (APA)
# ═══════════════════════════════════════════════════════════════════════════════
pdf.add_page()
pdf.section_title("Output F -- Supporting References (APA)", 1)

pdf.bold_text("Already cited in the GA-LLaMEA manuscript (in-text only, no new APA entry needed):")
pdf.body_text(
    "Sudholt, D. (2017). How crossover speeds up building block assembly in genetic algorithms. "
    "Evolutionary Computation, 25(2), 237-274. https://doi.org/10.1162/EVCO_a_00171\n\n"
    "van Stein, N., & Back, T. (2025). LLaMEA: A large language model evolutionary algorithm "
    "for automatically generating metaheuristics. IEEE Transactions on Evolutionary Computation, "
    "29, 2144-2157. https://doi.org/10.1109/TEVC.2024.3497793\n\n"
    "van Stein, N., Yin, H., Kononova, A. V., Back, T., & Ochoa, G. (2025). Behaviour space "
    "analysis of LLM-driven meta-heuristic discovery. arXiv preprint arXiv:2507.03605."
)

pdf.ln(4)
pdf.bold_text("NEW references added for Section 4.5 (provide full APA entry in manuscript reference list):")

pdf.body_text(
    "Liu, F., Tong, X., Yuan, M., Lin, X., Luo, F., Wang, Z., Lu, Z., & Zhang, Q. (2024). "
    "Evolution of heuristics: Towards efficient automatic algorithm design using large language "
    "model. In Proceedings of the 41st International Conference on Machine Learning "
    "(ICML 2024), Proceedings of Machine Learning Research, 235, 32683-32715. PMLR."
)
pdf.body_text(
    "Sun, L., & Li, K. (2020). Adaptive operator selection based on dynamic Thompson sampling "
    "for MOEA/D. In T. Back et al. (Eds.), Parallel Problem Solving from Nature -- PPSN XVI "
    "(Lecture Notes in Computer Science, Vol. 12270, pp. 271-284). Springer. "
    "https://doi.org/10.1007/978-3-030-58115-2_19"
)
pdf.body_text(
    "Tian, Y., Li, X., Ma, H., Zhang, X., Tan, K. C., & Jin, Y. (2022). Deep reinforcement "
    "learning based adaptive operator selection for evolutionary multi-objective optimization. "
    "IEEE Transactions on Emerging Topics in Computational Intelligence, 7(4), 1051-1064. "
    "https://doi.org/10.1109/TETCI.2022.3146882"
)

pdf.ln(4)
pdf.section_title("In-Text Citation Map", 2)
cite_map = [
    ["Location", "Claim Supported", "Citation"],
    ["4.5.1", "Exploitation dynamics associated with AOCC performance", "van Stein et al. (2025)"],
    ["4.5.1", "Feature-selection protocol (non-redundant features)", "van Stein et al. (2025)"],
    ["4.5.2", "Gold standard: focused exploitation-heavy profile", "van Stein et al. (2025)"],
    ["4.5.3", "EoH dual-population framework design", "Liu et al. (2024)"],
    ["4.5.3", "Baseline LLaMEA mutation-only refinement", "van Stein & Back (2025)"],
    ["4.5.4", "Crossover combines complementary strategies", "Sudholt (2017)"],
    ["4.5.4", "Discounted Thompson Sampling / AOS", "Sun & Li (2020); Tian et al. (2022)"],
]
cite_widths = [20, 90, 60]
pdf.add_table(cite_map[0], cite_map[1:], cite_widths)

# Save
pdf.output(os.path.join(os.path.dirname(__file__), OUT_PATH))
print(f"PDF saved to: {OUT_PATH}")
