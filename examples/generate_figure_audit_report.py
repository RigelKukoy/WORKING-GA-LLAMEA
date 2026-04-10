"""
Generate the Figure Audit Report PDF for GA-LLAMEA_MANUSCRIPT-Draft.docx.
Documents all figure naming errors, wrong number formats, missing captions,
and completely absent figures found in the Results and Discussion chapter.
Run: python generate_figure_audit_report.py
"""
from fpdf import FPDF
import os

OUT_PATH = "Figure_Audit_Report.pdf"

# ── Colour palette ───────────────────────────────────────────────────────────
RED    = (180, 30,  30)
AMBER  = (160, 90,   0)
GREEN  = (20,  100, 40)
BLUE   = (20,  60,  120)
GREY   = (80,  80,  80)

# ── PDF helper ───────────────────────────────────────────────────────────────
class Report(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*GREY)
        self.cell(0, 6, "GA-LLaMEA Manuscript  |  Figure Audit Report", align="C")
        self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def chapter_title(self, title):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(*BLUE)
        self.ln(4)
        self.cell(0, 9, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*BLUE)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    def section_title(self, title, color=BLUE):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(*color)
        self.ln(5)
        self.multi_cell(0, 6, title)
        self.ln(1)

    def body(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5.4, text)
        self.ln(2)

    def italic(self, text):
        self.set_font("Helvetica", "I", 10)
        self.set_text_color(60, 60, 60)
        self.multi_cell(0, 5.4, text)
        self.ln(2)

    def bold(self, text, color=(30, 30, 30)):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*color)
        self.multi_cell(0, 5.4, text)
        self.ln(1)

    def badge(self, label, color):
        """Print a small coloured label inline (starts on its own line)."""
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(255, 255, 255)
        self.set_fill_color(*color)
        self.cell(0, 6, f"  {label}  ", fill=True, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def search_tip(self, text):
        """Print a Ctrl+F search hint box."""
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(40, 40, 40)
        self.set_fill_color(240, 245, 255)
        self.set_draw_color(*BLUE)
        self.multi_cell(0, 5.4, f"  Ctrl+F: \"{text}\"", border=1, fill=True)
        self.ln(3)

    def fix_box(self, text):
        """Green fix instruction box."""
        self.set_font("Helvetica", "", 10)
        self.set_text_color(10, 60, 10)
        self.set_fill_color(230, 248, 230)
        self.set_draw_color(*GREEN)
        self.multi_cell(0, 5.4, f"  FIX: {text}", border=1, fill=True)
        self.ln(3)

    def add_table(self, headers, rows, col_widths=None):
        if col_widths is None:
            usable = self.w - self.l_margin - self.r_margin
            col_widths = [usable / len(headers)] * len(headers)
        self.set_font("Helvetica", "B", 8)
        self.set_fill_color(*BLUE)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, str(h), border=1, fill=True, align="C")
        self.ln()
        self.set_font("Helvetica", "", 8)
        self.set_text_color(30, 30, 30)
        for ri, row in enumerate(rows):
            fill = ri % 2 == 0
            self.set_fill_color(235, 240, 250) if fill else self.set_fill_color(255, 255, 255)
            for i, val in enumerate(row):
                self.cell(col_widths[i], 6, str(val), border=1, fill=fill, align="L")
            self.ln()
        self.ln(4)


# ═════════════════════════════════════════════════════════════════════════════
pdf = Report()
pdf.alias_nb_pages()
pdf.set_auto_page_break(auto=True, margin=20)
pdf.add_page()

# ── Cover / Intro ─────────────────────────────────────────────────────────────
pdf.chapter_title("Figure Audit Report")
pdf.bold("Document audited:  GA-LLAMEA_MANUSCRIPT-Draft.docx")
pdf.bold("Chapter audited:   Chapter IV -- Results and Discussion")
pdf.body(
    "This report documents every figure-related issue found in the Results and Discussion "
    "chapter of the GA-LLaMEA manuscript. Issues are grouped into four types:\n\n"
    "  ISSUE 1  Wrong caption title (caption does not match the figure it describes)\n"
    "  ISSUE 2  Wrong number format (text says 'Figure 10/11/12/13' instead of 'Figure 4.X')\n"
    "  ISSUE 3  Missing caption (figure cited in text but no caption line exists)\n"
    "  ISSUE 4  Completely absent (no in-text reference AND no caption)\n\n"
    "Each issue entry shows: the section where it occurs, a Ctrl+F search string to navigate "
    "there immediately, and the exact fix required."
)
pdf.italic(
    "Note: Word calculates page numbers by rendering. Because Section 4.3 (Behaviour Profile "
    "Analysis) was inserted after the TOC was written, all downstream page numbers in the TOC "
    "are now shifted. Use the Ctrl+F strings below to locate each issue precisely."
)

# ── Summary table ─────────────────────────────────────────────────────────────
pdf.section_title("Summary of All Issues", BLUE)
summary_headers = ["#", "Type", "Figure", "Section", "Status"]
summary_rows = [
    ["1", "Wrong caption title",   "Fig 4.4",  "4.3.1",  "HIGH PRIORITY"],
    ["2", "Wrong number format",   "Fig 4.5",  "4.3.2",  "Fix in text"],
    ["3", "Wrong number format",   "Fig 4.10", "4.6.1",  "Fix in text (x4)"],
    ["4", "Wrong number format",   "Fig 4.13", "4.6.2",  "Fix in text (x2)"],
    ["5", "Wrong number format",   "Fig 4.12", "4.6.2",  "Fix in text"],
    ["6", "Missing caption",       "Fig 4.2",  "4.2",    "Add caption"],
    ["7", "Missing caption",       "Fig 4.8",  "4.5",    "Add caption"],
    ["8", "Missing caption",       "Fig 4.12", "4.6.2",  "Add caption"],
    ["9", "Completely absent",     "Fig 4.3",  "4.2",    "Add ref + move caption"],
    ["10","Completely absent",     "Fig 4.9",  "4.5",    "Investigate"],
    ["11","Completely absent",     "Fig 4.11", "4.6.1",  "Add fig + caption + ref"],
]
summary_widths = [8, 42, 22, 20, 35]
pdf.add_table(summary_headers, summary_rows, summary_widths)

# ═════════════════════════════════════════════════════════════════════════════
# ISSUE 1
# ═════════════════════════════════════════════════════════════════════════════
pdf.add_page()
pdf.chapter_title("ISSUE 1 -- Wrong Caption Title  [HIGH PRIORITY]")

pdf.badge("SEVERITY: HIGH -- Caption describes completely wrong figure", RED)
pdf.section_title("Location:  Section 4.3.1  Metric Correlation and Feature Selection")
pdf.search_tip("Figure 4.4 Evolutionary Convergence")

pdf.bold("What is currently written:")
pdf.italic(
    "\"Figure 4.4 Evolutionary Convergence Trajectories: Mean Best AOCC over 100 Evaluation Steps\""
)
pdf.body(
    "This caption appears immediately after the correlation analysis paragraphs in Section 4.3.1. "
    "The body text directly above it reads:\n\n"
    "  \"A Pearson correlation analysis was first performed... (Figure 4.4)\"\n\n"
    "So the in-text reference uses Figure 4.4 for the CORRELATION HEATMAP. However, the "
    "caption title describes the AOCC convergence trajectories plot -- a completely different "
    "figure that belongs in Section 4.2 (Analysis of AOCC Trajectories and Fitness Distribution)."
)

pdf.bold("Root cause:")
pdf.body(
    "The AOCC trajectories caption was accidentally placed inside Section 4.3.1 instead of "
    "Section 4.2. It also inherited the wrong figure number (4.4 instead of 4.3). "
    "See Issue 9 for the corresponding missing reference in Section 4.2."
)

pdf.fix_box(
    "Change the caption to:\n"
    "  \"Figure 4.4 Behaviour Metrics Correlation Heatmap\"\n\n"
    "This matches the figure that should appear here: the 12x12 Pearson r correlation matrix "
    "(file: BehaviourMetrics.png from Result visualizations/BehaviourResults/)."
)

# ═════════════════════════════════════════════════════════════════════════════
# ISSUE 2
# ═════════════════════════════════════════════════════════════════════════════
pdf.add_page()
pdf.chapter_title("ISSUE 2 -- Wrong Number Format in Text (8 occurrences)")

pdf.badge("SEVERITY: MEDIUM -- Inconsistent figure numbering throughout chapter", AMBER)
pdf.body(
    "The manuscript uses 'Figure 4.X' format throughout Sections 4.1-4.4. However, in "
    "Sections 4.3.2, 4.6.1, and 4.6.2 the text drops to bare numbers ('Figure 10', "
    "'Figure 11', 'Figure 12', 'Figure 13'). These must all be corrected to the 4.X format."
)

pdf.section_title("2a  --  Section 4.3.2  Gold Standard Behaviour Profile")
pdf.search_tip("Figure 11 presents parallel")
pdf.body("Paragraph begins: \"Figure 11 presents parallel coordinate plots of all 1,442 algorithms...\"")
pdf.fix_box("Change \"Figure 11\" to \"Figure 4.5\"")

pdf.section_title("2b  --  Section 4.6.1  Effect of Generative Crossover  (4 occurrences)")
pdf.body("All four occurrences are in paragraphs discussing crossover ablation results.")

rows_2b = [
    ["Ctrl+F string",                         "Fix"],
    ["\"As summarized in Figure 10,\"",       "-> Figure 4.10"],
    ["\"shown in Figure 10, where multi\"",   "-> Figure 4.10"],
    ["\"in Figure 10 supports these\"",       "-> Figure 4.10"],
    ["\"in Figure 10 and is consistent\"",    "-> Figure 4.10"],
]
pdf.add_table(rows_2b[0], rows_2b[1:], [100, 67])

pdf.section_title("2c  --  Section 4.6.2  Effect of Population Initialization Size  (3 occurrences)")

rows_2c = [
    ["Ctrl+F string",                              "Fix"],
    ["\"As illustrated in Figure 13,\"",          "-> Figure 4.13"],
    ["\"shown in Figure 13. GA-LLaMEA-8\"",       "-> Figure 4.13"],
    ["\"in Figure 12 further supports\"",          "-> Figure 4.12"],
]
pdf.add_table(rows_2c[0], rows_2c[1:], [100, 67])

# ═════════════════════════════════════════════════════════════════════════════
# ISSUE 3
# ═════════════════════════════════════════════════════════════════════════════
pdf.add_page()
pdf.chapter_title("ISSUE 3 -- Missing Captions (figure cited in text, no caption exists)")

pdf.badge("SEVERITY: HIGH -- Figures will appear unlabelled in the final document", RED)
pdf.body(
    "The following figures are referenced in the body text but have no corresponding "
    "caption line anywhere in the document. Each needs a caption added immediately after "
    "the figure placeholder."
)

# 3a
pdf.section_title("3a  --  Figure 4.2  |  Section 4.2  Analysis of AOCC Trajectories")
pdf.search_tip("box plot in Fig. 4.2")
pdf.body(
    "Paragraph: \"As illustrated in the box plot in Fig. 4.2, the proposed framework "
    "maintains a significantly higher median fitness...\"\n\n"
    "The figure is referenced but no caption line ('Figure 4.2 ...') exists anywhere."
)
pdf.fix_box(
    "Add caption after the paragraph ending \"...but lower peak performance.\":\n"
    "  \"Figure 4.2 Final Fitness Distribution Box Plot of GA-LLaMEA, EoH, and Baseline LLaMEA\"\n\n"
    "Suggested image: the AOCC/fitness box plot comparing the three methods."
)

# 3b
pdf.section_title("3b  --  Figure 4.8  |  Section 4.5  Operator Selection Dynamics")
pdf.search_tip("shown in figure 4.8")
pdf.body(
    "Paragraph: \"The Operator selection, shown in figure 4.8, exhibits a clear adaptive "
    "pattern over time...\"\n\n"
    "No caption for Figure 4.8 exists. Note also the lowercase 'f' in 'figure 4.8' -- "
    "this should be capitalised to 'Figure 4.8' for consistency."
)
pdf.fix_box(
    "1. Capitalise: change \"figure 4.8\" to \"Figure 4.8\" in the sentence.\n"
    "2. Add caption after the last paragraph of Section 4.5 (ending "
    "\"...without relying on static scheduling strategies.\"):\n"
    "  \"Figure 4.8 Operator Selection Probabilities over Evaluation Budget\""
)

# 3c
pdf.section_title("3c  --  Figure 4.12  |  Section 4.6.2  Population Initialization Size")
pdf.search_tip("in Figure 12 further supports")
pdf.body(
    "Paragraph: \"The distribution of final fitness values in Figure 12 further supports "
    "this observation.\"\n\n"
    "After fixing the number to 'Figure 4.12' (Issue 2c), a caption for Figure 4.12 must "
    "also be added. Currently only Figure 4.13 has a caption in this section."
)
pdf.fix_box(
    "Add caption between the body text and the existing 'Figure 4.13' caption:\n"
    "  \"Figure 4.12 Convergence Trajectories: GA-LLaMEA-4 vs GA-LLaMEA-8\"\n\n"
    "This separates the convergence trajectory plot (4.12) from the box plot (4.13)."
)

# ═════════════════════════════════════════════════════════════════════════════
# ISSUE 4
# ═════════════════════════════════════════════════════════════════════════════
pdf.add_page()
pdf.chapter_title("ISSUE 4 -- Completely Absent Figures (no reference AND no caption)")

pdf.badge("SEVERITY: HIGH -- These figures are not present in the document at all", RED)
pdf.body(
    "The following figures have neither an in-text reference nor a caption. "
    "They are identified from gaps in the numbering sequence and from paragraphs "
    "that describe visual evidence without citing a figure."
)

# 4a
pdf.section_title("4a  --  Figure 4.3  |  Section 4.2  Analysis of AOCC Trajectories")
pdf.search_tip("GA-LLaMEA maintains a sustained and steep improvement trajectory")
pdf.body(
    "The paragraph beginning \"Convergence trajectories further highlight distinct anytime "
    "performance behaviors. GA-LLaMEA maintains a sustained and steep improvement "
    "trajectory...\" describes a convergence trajectory figure in detail but contains "
    "NO figure reference.\n\n"
    "This is almost certainly the same figure whose caption was accidentally placed in "
    "Section 4.3.1 (Issue 1: 'Figure 4.4 Evolutionary Convergence Trajectories...'). "
    "That caption belongs here, renumbered as Figure 4.3."
)
pdf.fix_box(
    "Step 1: Add '(Fig. 4.3)' at the end of the sentence:\n"
    "  \"...enabling efficient early-stage search followed by stable refinement (Fig. 4.3).\"\n\n"
    "Step 2: Remove the misplaced caption from Section 4.3.1 (see Issue 1).\n\n"
    "Step 3: Add the correctly titled caption here after the last paragraph of Section 4.2:\n"
    "  \"Figure 4.3 Evolutionary Convergence Trajectories: Mean Best AOCC over 100 Evaluation Steps\""
)

# 4b
pdf.section_title("4b  --  Figure 4.9  |  Section 4.5  Operator Selection Dynamics")
pdf.search_tip("4.5 Operator Selection Dynamics")
pdf.body(
    "Section 4.5 currently contains only one figure reference: Figure 4.8 (operator "
    "selection probabilities). The number gap between Fig 4.8 and Fig 4.10 (crossover "
    "ablation in Section 4.6.1) suggests Figure 4.9 is missing from Section 4.5.\n\n"
    "Operator selection sections typically require two visualisations: one showing "
    "operator selection probability over time, and one showing mean reward per operator "
    "across runs. If you have a second operator selection plot (e.g., reward distribution "
    "or cumulative selection counts), it should be inserted here as Figure 4.9.\n\n"
    "If no second figure exists for this section, then Figure 4.10 onward must be "
    "renumbered to close the gap (4.9, 4.10, 4.11, 4.12)."
)
pdf.fix_box(
    "Option A: Insert the second operator selection figure as Figure 4.9 with caption:\n"
    "  \"Figure 4.9 Mean Operator Reward over Evaluation Budget\"\n"
    "  Add in-text reference in the paragraph ending '...the effectiveness of generative "
    "recombination as a productive search mechanism.'\n\n"
    "Option B: If no second figure exists, renumber 4.10 -> 4.9, 4.11 -> 4.10, "
    "4.12 -> 4.11, 4.13 -> 4.12 and update all in-text references accordingly."
)

# 4c
pdf.section_title("4c  --  Figure 4.11  |  Section 4.6.1  Effect of Generative Crossover")
pdf.search_tip("distribution of final fitness values in Figure 10")
pdf.body(
    "Paragraph: \"Furthermore, the distribution of final fitness values in Figure 10 "
    "supports these findings. LLaMEA-Crossover3-NoRefine achieves the highest median "
    "with low variability...\"\n\n"
    "This sentence describes a BOX PLOT of final fitness values, but the only captioned "
    "figure in this section is 'Figure 4.10 Fitness Convergence Plot (Crossover Ablation)' "
    "which is a CONVERGENCE TRAJECTORY plot -- a different visualisation entirely.\n\n"
    "The file 'Crossover-ablation-boxplot.png' in the examples folder is this missing "
    "figure. It must be inserted as Figure 4.11 with its own caption and its own "
    "in-text reference (currently incorrectly sharing 'Figure 10' with Fig 4.10)."
)
pdf.fix_box(
    "Step 1: Change the in-text reference for the box plot:\n"
    "  From: \"...distribution of final fitness values in Figure 10 supports...\"\n"
    "  To:   \"...distribution of final fitness values in Figure 4.11 supports...\"\n\n"
    "Step 2: Add caption after Figure 4.10 caption and before Section 4.6.2:\n"
    "  \"Figure 4.11 Fitness Distribution Box Plot (Crossover Ablation)\"\n\n"
    "Step 3: Insert the image 'Crossover-ablation-boxplot.png' at that location.\n\n"
    "Also check whether the sentence in paragraph 4 of Section 4.6.1 -- "
    "\"in Figure 10 and is consistent...\" -- refers to the convergence plot (4.10) "
    "or the box plot (4.11), and update accordingly."
)

# ═════════════════════════════════════════════════════════════════════════════
# Quick Reference
# ═════════════════════════════════════════════════════════════════════════════
pdf.add_page()
pdf.chapter_title("Quick-Fix Checklist")

pdf.body(
    "Work through issues in the order below. Issues 1 and 4a are linked -- fix them together."
)

checklist_headers = ["Priority", "Action", "Section", "Ctrl+F to locate"]
checklist_rows = [
    ["1 (HIGH)",  "Rename Figure 4.4 caption to 'Behaviour Metrics Correlation Heatmap'",
     "4.3.1",     "Figure 4.4 Evolutionary"],
    ["2 (HIGH)",  "Remove misplaced AOCC Trajectories caption from 4.3.1",
     "4.3.1",     "Figure 4.4 Evolutionary"],
    ["3 (HIGH)",  "Add Figure 4.3 caption + ref in Section 4.2 (AOCC Trajectories)",
     "4.2",       "GA-LLaMEA maintains a sustained"],
    ["4 (HIGH)",  "Add missing Figure 4.2 caption (fitness box plot)",
     "4.2",       "box plot in Fig. 4.2"],
    ["5 (HIGH)",  "Add Figure 4.11 (crossover box plot) + fix its in-text ref",
     "4.6.1",     "distribution of final fitness values in Figure 10"],
    ["6 (HIGH)",  "Add missing Figure 4.8 caption (operator selection)",
     "4.5",       "shown in figure 4.8"],
    ["7 (HIGH)",  "Capitalise 'figure 4.8' -> 'Figure 4.8'",
     "4.5",       "shown in figure 4.8"],
    ["8 (HIGH)",  "Add missing Figure 4.12 caption (pop. init convergence)",
     "4.6.2",     "in Figure 12 further supports"],
    ["9 (MED)",   "Fix 'Figure 11' -> 'Figure 4.5'",
     "4.3.2",     "Figure 11 presents parallel"],
    ["10 (MED)",  "Fix 4x 'Figure 10' -> 'Figure 4.10'",
     "4.6.1",     "As summarized in Figure 10"],
    ["11 (MED)",  "Fix 2x 'Figure 13' -> 'Figure 4.13'",
     "4.6.2",     "As illustrated in Figure 13"],
    ["12 (MED)",  "Fix 'Figure 12' -> 'Figure 4.12'",
     "4.6.2",     "in Figure 12 further supports"],
    ["13 (LOW)",  "Investigate Figure 4.9 gap in Section 4.5",
     "4.5",       "4.5 Operator Selection Dynamics"],
]
checklist_widths = [20, 75, 18, 57]
pdf.add_table(checklist_headers, checklist_rows, checklist_widths)

pdf.ln(4)
pdf.section_title("Figure Sequence After All Fixes", GREEN)
sequence_headers = ["Figure", "Caption", "Section", "Image file"]
sequence_rows = [
    ["4.1",  "Comparative Elo Ratings",                         "4.1",   "Elo figure"],
    ["4.2",  "Final Fitness Distribution Box Plot",             "4.2",   "Fitness box plot"],
    ["4.3",  "Evolutionary Convergence Trajectories (AOCC)",    "4.2",   "AOCC trajectories"],
    ["4.4",  "Behaviour Metrics Correlation Heatmap",           "4.3.1", "BehaviourMetrics.png"],
    ["4.5",  "Behaviour Profile of All Generated Algorithms",   "4.3.2", "Behaviour Profile (All Algorithms, Quartile Colour).png"],
    ["4.6",  "Top-100 Behaviour Profiles per Method",           "4.3.3", "Top-100 Algorithms per Method.png"],
    ["4.7",  "Code Evolution Graphs (5 runs per method)",       "4.4",   "CEG.png"],
    ["4.8",  "Operator Selection Probabilities",                "4.5",   "Operator selection figure"],
    ["4.9",  "Mean Operator Reward (if available)",             "4.5",   "Second operator fig or renumber"],
    ["4.10", "Fitness Convergence Plot (Crossover Ablation)",   "4.6.1", "crossover-boxplot.png"],
    ["4.11", "Fitness Distribution Box Plot (Crossover)",       "4.6.1", "Crossover-ablation-boxplot.png"],
    ["4.12", "Convergence Trajectories (Pop. Init Ablation)",   "4.6.2", "Pop. init convergence"],
    ["4.13", "Fitness/AOCC Box Plot (Pop. Init Ablation)",      "4.6.2", "populationinit-boxplot.png"],
]
sequence_widths = [14, 65, 18, 73]
pdf.add_table(sequence_headers, sequence_rows, sequence_widths)

# ── Save ─────────────────────────────────────────────────────────────────────
pdf.output(os.path.join(os.path.dirname(__file__), OUT_PATH))
print(f"PDF saved to: {OUT_PATH}")
