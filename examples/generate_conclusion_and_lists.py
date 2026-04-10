"""
Generate: Updated Conclusion + All Document Lists PDF
- Updated Chapter V Conclusion (with Behaviour Profile Analysis addition)
- Table of Contents (updated with current section numbering)
- List of Tables
- List of Figures (all 13 figures with correct captions)
- List of Equations
- List of Algorithms
- New APA citations for added references

Run from: WORKING-GA-LLAMEA/examples/
"""
import os
from fpdf import FPDF

OUT_PATH = "Conclusion_and_Lists_UPDATE.pdf"

# ── Colours ──────────────────────────────────────────────────────────────────
DARK_BLUE  = (20,  60,  120)
MID_BLUE   = (40,  90,  160)
LIGHT_BLUE = (210, 225, 245)
GREEN      = (20,  100,  40)
GREY       = (80,  80,   80)
DARK       = (30,  30,   30)
NOTE_BG    = (255, 250, 220)

# ── PDF class ─────────────────────────────────────────────────────────────────
class Doc(FPDF):

    def header(self):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*GREY)
        self.cell(0, 6, "GA-LLaMEA Manuscript  |  Conclusion Update + Document Lists", align="C")
        self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    # -- Title helpers ---------------------------------------------------------
    def chapter_title(self, text):
        self.set_font("Helvetica", "B", 15)
        self.set_text_color(*DARK_BLUE)
        self.ln(4)
        self.cell(0, 9, text, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*DARK_BLUE)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(5)

    def section_head(self, text, size=11):
        self.set_font("Helvetica", "B", size)
        self.set_text_color(*MID_BLUE)
        self.ln(4)
        self.multi_cell(0, 6, text)
        self.ln(2)

    def body(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*DARK)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def italic(self, text):
        self.set_font("Helvetica", "I", 10)
        self.set_text_color(60, 60, 60)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def bold(self, text):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*DARK)
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def note_box(self, text):
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(80, 60, 0)
        self.set_fill_color(*NOTE_BG)
        self.set_draw_color(180, 140, 0)
        self.multi_cell(0, 5.5, text, border=1, fill=True)
        self.ln(3)

    def new_ref_box(self, text):
        self.set_font("Helvetica", "", 9)
        self.set_text_color(10, 60, 10)
        self.set_fill_color(230, 248, 230)
        self.set_draw_color(*GREEN)
        self.multi_cell(0, 5.5, text, border=1, fill=True)
        self.ln(3)

    def add_table(self, headers, rows, col_widths):
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(*DARK_BLUE)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, h, border=1, fill=True, align="C")
        self.ln()
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*DARK)
        for ri, row in enumerate(rows):
            self.set_fill_color(*LIGHT_BLUE) if ri % 2 == 0 else self.set_fill_color(255, 255, 255)
            for i, val in enumerate(row):
                align = "C" if i == len(row) - 1 else "L"
                self.cell(col_widths[i], 6, str(val), border=1, fill=(ri % 2 == 0), align=align)
            self.ln()
        self.ln(4)

    # -- Two-column TOC row ----------------------------------------------------
    def toc_row(self, label, page, indent=0):
        usable = self.w - self.l_margin - self.r_margin
        indent_mm = indent * 4
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*DARK)
        self.set_x(self.l_margin + indent_mm)
        text_w = usable - indent_mm - 15
        # dots
        self.cell(text_w, 6, label, align="L")
        self.set_font("Helvetica", "B", 10)
        self.cell(15, 6, str(page), align="R")
        self.ln()

    def toc_row_bold(self, label, page, indent=0):
        usable = self.w - self.l_margin - self.r_margin
        indent_mm = indent * 4
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*DARK_BLUE)
        self.set_x(self.l_margin + indent_mm)
        text_w = usable - indent_mm - 15
        self.cell(text_w, 7, label, align="L")
        self.cell(15, 7, str(page), align="R")
        self.ln()


# =============================================================================
pdf = Doc()
pdf.alias_nb_pages()
pdf.set_auto_page_break(auto=True, margin=20)
pdf.add_page()

# =============================================================================
# SECTION 1 -- INTRODUCTION / HOW TO USE THIS DOCUMENT
# =============================================================================
pdf.chapter_title("About This Document")
pdf.body(
    "This document contains two deliverables for the GA-LLaMEA manuscript:\n\n"
    "  Part 1 -- Updated Chapter V Conclusion\n"
    "  A new paragraph on Behaviour Profile Analysis has been added after the existing\n"
    "  convergence/CEG paragraph. The full copy-paste-ready conclusion is provided.\n\n"
    "  Part 2 -- Updated Document Lists\n"
    "  Corrected and complete versions of the Table of Contents, List of Tables,\n"
    "  List of Figures, List of Equations, and List of Algorithms.\n\n"
    "  Note on page numbers:\n"
    "  Pages for Chapter I-III and Sections 4.1-4.2 are confirmed from the existing TOC.\n"
    "  Pages for Section 4.3 onward are estimated based on section length after the\n"
    "  insertion of the new Behaviour Profile Analysis section (~+4 pages shift).\n"
    "  Verify and update page numbers in Word after all edits are finalized."
)
pdf.note_box(
    "  Quick update tip: In Word, press Ctrl+A (select all), then F9 (update fields).\n"
    "  This automatically recalculates page numbers in the TOC and all list fields."
)

# =============================================================================
# SECTION 2 -- UPDATED CONCLUSION
# =============================================================================
pdf.add_page()
pdf.chapter_title("Part 1 -- Updated Chapter V: Conclusion")
pdf.note_box(
    "  The paragraph marked [NEW] below is the addition based on Behaviour Profile Analysis\n"
    "  findings from Section 4.3. Insert it after the paragraph ending '...van Stein et al.,\n"
    "  2025; van Stein et al., 2025b).' and before the theoretical perspective paragraph."
)
pdf.ln(2)

pdf.section_head("Chapter V")
pdf.section_head("CONCLUSION", 13)
pdf.ln(2)

# Para 1 -- existing
pdf.body(
    "This study presented GA-LLaMEA, a generative-adaptive evolutionary framework developed "
    "to address three structural limitations in current Large Language Model driven Automated "
    "Algorithm Design systems: mutation-centric evolution, static operator scheduling, and "
    "semantically weak recombination (Liu et al., 2024; Wu et al., 2024). By integrating "
    "Inspiration-Based Generative Crossover with Discounted Thompson Sampling for adaptive "
    "operator selection, the framework introduces both structural diversity and dynamic control "
    "into LLM-mediated optimization. This approach directly responds to recent calls for more "
    "structured and adaptive LLM-assisted evolutionary systems (van Stein et al., 2025; Wu et "
    "al., 2024). Empirical evaluation on the five-dimensional MA-BBOB benchmark demonstrated "
    "consistent and statistically stable improvements across competitive performance, convergence "
    "behavior, and structural integrity. GA-LLaMEA achieved the highest mean Elo rating among "
    "all compared frameworks, outperforming LLaMEA-Prompt5 and Evolution of Heuristics (Liu et "
    "al., 2024)."
)

# Para 2 -- existing (behavioral + convergence + CEG)
pdf.body(
    "More importantly, its superiority was not limited to final solution quality. Behavioral "
    "metrics revealed a stronger exploitation profile, including the lowest average "
    "distance-to-best and the highest intensification ratio, indicating effective local "
    "refinement once promising regions were identified. This aligns with established "
    "exploration-exploitation theory, which emphasizes dynamic balancing across optimization "
    "stages (Berger-Tal et al., 2014). Convergence analysis further showed that GA-LLaMEA "
    "produced sustained incremental improvements with minimal stagnation. The framework "
    "exhibited the highest success rate, the shortest longest no-improvement streak, and the "
    "lowest last-improvement fraction, demonstrating reliability throughout the optimization "
    "process rather than dependence on large sporadic improvement steps. These findings "
    "reinforce pattern-based perspectives in heuristic optimization, where adaptation and "
    "controlled intensification are essential design components (Damasevicius, 2025). Structural "
    "analysis through Code Evolution Graphs provided additional insight into genealogical "
    "stability. Unlike mutation-dominant baselines that exhibited unstable complexity growth and "
    "token expansion, GA-LLaMEA maintained interconnected lineage structures and controlled "
    "code length. The guided conceptual transfer mechanism successfully prevented blind syntactic "
    "recombination, preserving semantic coherence while enabling strategic building-block "
    "synthesis. This directly addresses concerns regarding opacity and uncontrolled complexity "
    "in LLM-driven algorithm generation (van Stein et al., 2025; van Stein et al., 2025b)."
)

# Para 3 -- NEW: Behaviour Profile Analysis
pdf.set_font("Helvetica", "B", 10)
pdf.set_text_color(*GREEN)
pdf.cell(0, 6, "[NEW PARAGRAPH -- ADD HERE]", new_x="LMARGIN", new_y="NEXT")
pdf.set_text_color(*DARK)
pdf.ln(1)
pdf.body(
    "Behaviour-space analysis, conducted following the methodology of van Stein et al. (2025), "
    "provided further mechanistic evidence for GA-LLaMEA's superiority. Profiling all 1,442 "
    "generated algorithms across 11 behavioural metrics revealed a Q4 gold standard profile "
    "characterised by low exploration percentage (4.03%), high intensification ratio (0.9395), "
    "near-maximal convergence rate (0.9962), and sustained late-stage improvement. GA-LLaMEA "
    "aligned more closely with this profile than either baseline, placing 36.3% of its generated "
    "algorithms (177 out of 488) in the top-performance quartile, compared to 21.0% for Baseline "
    "LLaMEA and 17.6% for EoH. This demonstrates that GA-LLaMEA does not merely produce isolated "
    "high-quality solutions but systematically navigates the algorithm search space toward the "
    "gold standard behavioural region. The Discounted Thompson Sampling controller was empirically "
    "shown to progressively concentrate selection probability on the crossover and simplify "
    "operators, both of which consistently produced higher mean AOCC rewards than random "
    "generation across all five independent runs (Sun & Li, 2020; Tian et al., 2022). Mean "
    "operator reward analysis (Figure 4.9) confirmed that crossover maintained the highest "
    "reward (mean AOCC 0.59 at generation 1, rising to 0.79 by generation 12), while the "
    "random-new operator remained persistently low throughout, validating the adaptive control "
    "decisions of the DTS mechanism. Together, these findings establish that the performance "
    "advantage of GA-LLaMEA is not incidental but rooted in a coherent exploitation-focused "
    "search dynamic, lending behavioural interpretability to what would otherwise be treated "
    "as a black-box performance gain."
)

# Para 4 -- existing (theoretical)
pdf.body(
    "From a theoretical perspective, these findings align with established principles in "
    "evolutionary computation. Recombination facilitates efficient building-block assembly, "
    "while adaptive operator control is essential in non-stationary environments (Berger-Tal "
    "et al., 2014; Wu et al., 2024). Furthermore, integrating LLM reasoning into evolutionary "
    "frameworks reflects the broader paradigm of LLM-enhanced evolutionary computation (Wu et "
    "al., 2024) and LLM-guided evolutionary synthesis (Sadikov, 2025)."
)

# Para 5 -- existing (overall)
pdf.body(
    "Overall, GA-LLaMEA demonstrates that semantic-aware crossover combined with adaptive "
    "operator selection significantly enhances robustness, convergence reliability, and "
    "generalization performance in automated algorithm design. By coupling generative discovery "
    "with adaptive control, structural monitoring, and behaviour-space interpretability, the "
    "framework contributes toward the emerging vision of explainable and performance-aware "
    "automated algorithm design (van Stein et al., 2025b)."
)

pdf.ln(3)
pdf.section_head("New In-Text Citations Used in the Added Paragraph")
pdf.new_ref_box(
    "  Sun, L., & Li, K. (2020). Adaptive operator selection based on dynamic Thompson\n"
    "  sampling for MOEA/D. In Parallel Problem Solving from Nature -- PPSN XVI\n"
    "  (LNCS Vol. 12270, pp. 271-284). Springer.\n"
    "  https://doi.org/10.1007/978-3-030-58115-2_19\n"
    "  [Already added to manuscript reference list -- no action needed]\n\n"
    "  Tian, Y., Li, X., Ma, H., Zhang, X., Tan, K. C., & Jin, Y. (2022). Deep reinforcement\n"
    "  learning based adaptive operator selection for evolutionary multi-objective optimization.\n"
    "  IEEE Transactions on Emerging Topics in Computational Intelligence, 7(4), 1051-1064.\n"
    "  https://doi.org/10.1109/TETCI.2022.3146882\n"
    "  [Already added to manuscript reference list -- no action needed]"
)

# =============================================================================
# SECTION 3 -- TABLE OF CONTENTS
# =============================================================================
pdf.add_page()
pdf.chapter_title("Part 2 -- Updated Table of Contents")
pdf.note_box(
    "  Pages for Sections 4.3 onward are estimated (+4 pages from new Behaviour Profile\n"
    "  Analysis section). Use Ctrl+A then F9 in Word to auto-update all page numbers."
)
pdf.ln(2)

pdf.bold("                                                                    PAGE")
pdf.ln(1)

pdf.toc_row_bold("TITLE PAGE", "i")
pdf.toc_row_bold("APPROVAL PAGE", "ii")
pdf.toc_row_bold("TABLE OF CONTENTS", "iii")
pdf.toc_row_bold("LIST OF TABLES", "iv")
pdf.toc_row_bold("LIST OF FIGURES", "v")
pdf.toc_row_bold("LIST OF EQUATIONS", "vi")
pdf.toc_row_bold("LIST OF ALGORITHMS", "vii")
pdf.ln(2)

pdf.toc_row_bold("CHAPTER I  INTRODUCTION", 1)
pdf.toc_row("1.1  Background of the Study", 3, 1)
pdf.toc_row("1.2  Statement of the Problem", 5, 1)
pdf.toc_row("1.2.1  Mutation-Centric Optimization Limitation", 6, 2)
pdf.toc_row("1.2.2  Ineffective Recombination Mechanisms", 6, 2)
pdf.toc_row("1.2.3  Lack of Adaptive Operator Selection", 7, 2)
pdf.toc_row("1.3  Significance of the Study", 8, 1)
pdf.toc_row("1.4  Objectives of the Study", 10, 1)
pdf.toc_row("1.4.1  Develop the GA-LLaMEA Framework", 10, 2)
pdf.toc_row("1.4.2  Implement Generative Crossover Mechanisms", 10, 2)
pdf.toc_row("1.4.3  Validate the Framework using MA-BBOB Benchmarking", 10, 2)
pdf.toc_row("1.5  Scope and Limitation", 11, 1)
pdf.toc_row("1.5.1  Scope", 11, 2)
pdf.toc_row("1.5.2  Limitation", 12, 2)
pdf.ln(2)

pdf.toc_row_bold("CHAPTER II  REVIEW OF LITERATURE", 13)
pdf.toc_row("2.1  Theoretical Foundations of Evolutionary Computation", 13, 1)
pdf.toc_row("2.1.1  Exploration-Exploitation Trade-off", 14, 2)
pdf.toc_row("2.1.2  Building Block Hypothesis and Recombination", 14, 2)
pdf.toc_row("2.1.3  No Free Lunch Theorems", 16, 2)
pdf.toc_row("2.2  Evolution of Automated Algorithm Design (AAD)", 17, 1)
pdf.toc_row("2.2.1  Genetic Programming and Symbolic Regression", 17, 2)
pdf.toc_row("2.2.2  Hyper-Heuristics: Selection vs Generation", 18, 2)
pdf.toc_row("2.3  Large Language Models as Optimization Agents", 19, 1)
pdf.toc_row("2.3.1  Emergent Reasoning and In-Context Learning", 19, 2)
pdf.toc_row("2.3.2  LLM-Driven Automated Algorithm Design", 20, 2)
pdf.toc_row("2.3.3  State-of-the-Art Frameworks", 20, 2)
pdf.toc_row("2.4  The Deficit of Mutation-Centric Search", 22, 1)
pdf.toc_row("2.5  From Syntactic Splicing to Generative Crossover", 22, 1)
pdf.toc_row("2.6  Adaptive Operator Selection in Non-Stationary Environments", 23, 1)
pdf.toc_row("2.6.1  The Limitation of Static Control Schedules", 24, 2)
pdf.toc_row("2.6.2  Reinforcement Learning and Bandit Approaches", 24, 2)
pdf.toc_row("2.7  Search Landscape and Behavioral Analysis", 25, 1)
pdf.toc_row("2.7.1  Deception and the Fitness Trap", 25, 2)
pdf.toc_row("2.7.2  Behavioral Diversity and Search Trajectories", 25, 2)
pdf.toc_row("2.8  Research Gap and Synthesis", 26, 1)
pdf.ln(2)

pdf.toc_row_bold("CHAPTER III  METHODOLOGY", 27)
pdf.toc_row("3.1  Architectural Overview and Framework Comparison", 27, 1)
pdf.toc_row("3.1.1  Evolution of Heuristics (EoH)", 28, 2)
pdf.toc_row("3.1.2  Baseline LLaMEA Architecture", 29, 2)
pdf.toc_row("3.1.3  Proposed GA-LLaMEA Framework", 30, 2)
pdf.toc_row("3.2  Discounted Thompson Sampling", 31, 1)
pdf.toc_row("3.2.1  Discounted Statistics", 32, 2)
pdf.toc_row("3.2.2  Recursive Update Mechanism", 33, 2)
pdf.toc_row("3.2.3  Posterior Estimation", 34, 2)
pdf.toc_row("3.2.4  Posterior Sampling and Operator Selection", 34, 2)
pdf.toc_row("3.2.5  Practical Exploration Enhancements", 35, 2)
pdf.toc_row("3.2.6  Interpretation of Parameters", 35, 2)
pdf.toc_row("3.3  Reward System", 36, 1)
pdf.toc_row("3.4  Genetic Operators", 37, 1)
pdf.toc_row("3.5  Experimental Setup", 38, 1)
pdf.toc_row("3.6  Evaluation Metrics", 40, 1)
pdf.ln(2)

pdf.toc_row_bold("CHAPTER IV  RESULTS AND DISCUSSION", 43)
pdf.toc_row("4.1  Comparative Analysis of Elo Ratings", 43, 1)
pdf.toc_row("4.2  Analysis of AOCC Trajectories and Fitness Distribution", 45, 1)
pdf.toc_row("4.3  Behaviour Profile Analysis", "~47", 1)
pdf.toc_row("4.3.1  Metric Correlation and Feature Selection", "~47", 2)
pdf.toc_row("4.3.2  Gold Standard Behaviour Profile", "~49", 2)
pdf.toc_row("4.3.3  Comparative Behaviour Profiles of EoH, LLaMEA, and GA-LLaMEA", "~51", 2)
pdf.toc_row("4.3.4  Behavioural Interpretation of Performance Differences", "~53", 2)
pdf.toc_row("4.4  CEG Behaviour", "~55", 1)
pdf.toc_row("4.5  Operator Selection Dynamics", "~57", 1)
pdf.toc_row("4.6  Ablation Studies", "~59", 1)
pdf.toc_row("4.6.1  Effect of Generative Crossover", "~59", 2)
pdf.toc_row("4.6.2  Effect of Population Initialization Size", "~61", 2)
pdf.ln(2)

pdf.toc_row_bold("CHAPTER V  CONCLUSION", "~63")
pdf.toc_row_bold("CHAPTER VI  RECOMMENDATION", "~65")
pdf.toc_row_bold("REFERENCES", "~67")

# =============================================================================
# SECTION 4 -- LIST OF TABLES
# =============================================================================
pdf.add_page()
pdf.chapter_title("List of Tables")
pdf.ln(2)
pdf.bold("TABLE                                                                   PAGE")
pdf.ln(1)

tables = [
    ("Table 1",
     "Median Behaviour Metrics per Method vs. Q4 Gold Standard",
     "~52"),
]
for num, caption, pg in tables:
    pdf.toc_row(f"{num}   {caption}", pg, 0)

pdf.ln(4)
pdf.note_box(
    "  Only one table is present in the current manuscript body.\n"
    "  If additional tables are added, insert them here in order of appearance."
)

# =============================================================================
# SECTION 5 -- LIST OF FIGURES
# =============================================================================
pdf.add_page()
pdf.chapter_title("List of Figures")
pdf.ln(2)
pdf.bold("FIGURE                                                                  PAGE")
pdf.ln(1)

figures = [
    ("Figure 4.1",
     "Comparative Elo Ratings of GA-LLaMEA, EoH, and Baseline LLaMEA on MA-BBOB Instances",
     "~44"),
    ("Figure 4.2",
     "Final Fitness Distribution Box Plot of GA-LLaMEA, EoH, and Baseline LLaMEA",
     "~45"),
    ("Figure 4.3",
     "Evolutionary Convergence Trajectories: Mean Best AOCC over 100 Evaluation Steps",
     "~46"),
    ("Figure 4.4",
     "Behaviour Metrics Correlation Heatmap",
     "~48"),
    ("Figure 4.5",
     "Behaviour Profile of All Generated Algorithms (Quartile Colour)",
     "~50"),
    ("Figure 4.6",
     "Top-100 Behaviour Profiles per Method",
     "~52"),
    ("Figure 4.7",
     "Code Evolution Graphs (CEGs) across Five Runs per Method",
     "~56"),
    ("Figure 4.8",
     "Operator Selection Probabilities over Evaluation Budget",
     "~57"),
    ("Figure 4.9",
     "Mean Operator Reward over Evaluation Budget",
     "~58"),
    ("Figure 4.10",
     "Fitness Convergence Plot (Crossover Ablation)",
     "~60"),
    ("Figure 4.11",
     "Fitness Distribution Box Plot (Crossover Ablation)",
     "~61"),
    ("Figure 4.12",
     "Convergence Trajectories: GA-LLaMEA-4 vs GA-LLaMEA-8",
     "~62"),
    ("Figure 4.13",
     "Fitness/AOCC Box Plot (Population Initialization Size Ablation)",
     "~63"),
]

for num, caption, pg in figures:
    pdf.toc_row(f"{num}   {caption}", pg, 0)

pdf.ln(4)
pdf.note_box(
    "  Figures 4.2, 4.3, 4.8, 4.10, 4.11, and 4.12 still require caption lines in the\n"
    "  manuscript body. See Figure_Audit_Report.pdf for exact fix instructions.\n"
    "  Figure 4.9 has been generated and saved as Figure_4_9_Mean_Operator_Reward.png."
)

# =============================================================================
# SECTION 6 -- LIST OF EQUATIONS
# =============================================================================
pdf.add_page()
pdf.chapter_title("List of Equations")
pdf.ln(2)
pdf.bold("EQUATION                                                                PAGE")
pdf.ln(1)

equations = [
    ("Equation 3-1",  "Discounted Selection Count",                     "~32"),
    ("Equation 3-2",  "Discounted Cumulative Reward",                   "~32"),
    ("Equation 3-3",  "Recursive Discounting",                          "~33"),
    ("Equation 3-4",  "Observation Incorporation",                      "~34"),
    ("Equation 3-5",  "Posterior Mean Estimate",                        "~34"),
    ("Equation 3-6",  "Posterior Variance Estimate",                    "~34"),
    ("Equation 3-7",  "Independent Posterior Sampling",                 "~35"),
    ("Equation 3-8",  "Greedy Operator Selection",                      "~35"),
    ("Equation 3-9",  "Standard Thompson Sampling Policy",              "~35"),
    ("Equation 3-10", "Epsilon-Greedy Exploration Floor",               "~35"),
    ("Equation 3-11", "Preserved Posterior Update",                     "~36"),
    ("Equation 3-12", "Validity-Conditioned Reward Function",           "~36"),
    ("Equation 3-13", "Bandit Expected Reward Estimation",              "~37"),
    ("Equation 3-14", "Area Over the Convergence Curve (AOCC)",        "~40"),
]

for num, caption, pg in equations:
    pdf.toc_row(f"{num}   {caption}", pg, 0)

# =============================================================================
# SECTION 7 -- LIST OF ALGORITHMS
# =============================================================================
pdf.ln(6)
pdf.section_head("List of Algorithms")
pdf.ln(2)
pdf.bold("ALGORITHM                                                               PAGE")
pdf.ln(1)

algorithms = [
    ("Algorithm 1", "Overall Architecture of the GA-LLaMEA Framework", "~40"),
]
for num, caption, pg in algorithms:
    pdf.toc_row(f"{num}   {caption}", pg, 0)

# =============================================================================
# SECTION 8 -- FIGURES STILL NEEDING CAPTIONS (quick reference)
# =============================================================================
pdf.add_page()
pdf.chapter_title("Quick Reference -- Remaining Manuscript Actions")

pdf.section_head("Figures still needing caption lines in the .docx")
headers = ["Figure", "Caption to add", "Insert location"]
rows = [
    ["Fig 4.2",  "Final Fitness Distribution Box Plot of GA-LLaMEA, EoH, and Baseline LLaMEA",
     "After para ending '...lower peak performance.' in Section 4.2"],
    ["Fig 4.3",  "Evolutionary Convergence Trajectories: Mean Best AOCC over 100 Eval. Steps",
     "After last paragraph of Section 4.2 (move from Section 4.3.1)"],
    ["Fig 4.8",  "Operator Selection Probabilities over Evaluation Budget",
     "After last para of Section 4.5 (ending '...static scheduling strategies.')"],
    ["Fig 4.9",  "Mean Operator Reward over Evaluation Budget",
     "After Fig 4.8 caption in Section 4.5 (image: Figure_4_9_Mean_Operator_Reward.png)"],
    ["Fig 4.10", "Fitness Convergence Plot (Crossover Ablation)",
     "After crossover ablation body text in Section 4.6.1"],
    ["Fig 4.11", "Fitness Distribution Box Plot (Crossover Ablation)",
     "After Fig 4.10 caption -- image: Crossover-ablation-boxplot.png"],
    ["Fig 4.12", "Convergence Trajectories: GA-LLaMEA-4 vs GA-LLaMEA-8",
     "Before Fig 4.13 caption in Section 4.6.2"],
]
col_widths = [18, 80, 72]
pdf.add_table(headers, rows, col_widths)

pdf.section_head("In-text reference fixes still needed in the .docx")
fix_headers = ["Ctrl+F to locate", "Current (wrong)", "Should be"]
fix_rows = [
    ["\"shown in figure 4.8\"",          "figure 4.8 (lowercase)",   "Figure 4.8"],
    ["\"As summarized in Figure 10,\"",  "Figure 10",                "Figure 4.10"],
    ["\"shown in Figure 10, where\"",    "Figure 10",                "Figure 4.10"],
    ["\"in Figure 10 supports these\"",  "Figure 10",                "Figure 4.10"],
    ["\"in Figure 10 and is\"",          "Figure 10",                "Figure 4.10"],
    ["\"illustrated in Figure 13,\"",    "Figure 13",                "Figure 4.13"],
    ["\"shown in Figure 13. GA\"",       "Figure 13",                "Figure 4.13"],
    ["\"in Figure 12 further\"",         "Figure 12",                "Figure 4.12"],
]
fix_widths = [60, 40, 40]
pdf.add_table(fix_headers, fix_rows, fix_widths)

pdf.section_head("TOC section name to update")
pdf.body(
    "The Table of Contents entry for Section 4.3 currently reads:\n"
    "  '4.3  Evaluation of Behavioural Metrics'\n\n"
    "Update to:\n"
    "  '4.3  Behaviour Profile Analysis'\n\n"
    "Also update subsections 4.3.1-4.3.4 and renumber 4.4-4.7 to 4.4-4.6 as reflected\n"
    "in this document's updated Table of Contents above."
)

# =============================================================================
pdf.output(os.path.join(os.path.dirname(__file__), OUT_PATH))
print(f"PDF saved to: {OUT_PATH}")
