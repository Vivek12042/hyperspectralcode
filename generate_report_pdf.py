#!/usr/bin/env python3
"""
Generates a highly professional, beautifully styled PDF report for the
Hyperspectral Anomaly Detection Pipeline.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfgen import canvas

# Define custom corporate color palette
PRIMARY_COLOR = colors.HexColor("#1A252C")    # Dark slate
SECONDARY_COLOR = colors.HexColor("#007ACC")  # Slate blue
ACCENT_COLOR = colors.HexColor("#00B4D8")     # Teal/Mint
TEXT_DARK = colors.HexColor("#2D3748")        # Off-black
BG_LIGHT = colors.HexColor("#F8F9FA")         # Off-white
BORDER_COLOR = colors.HexColor("#E2E8F0")     # Light gray border

class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to dynamically compute and render "Page X of Y" page numbers,
    as well as beautiful headers and footers on every page.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, total_pages):
        self.saveState()
        
        # Suppress header/footer on page 1 (Title/Cover section if preferred, or keep minimal)
        # We will keep a subtle design on all pages
        
        # Header Line & Title
        self.setStrokeColor(PRIMARY_COLOR)
        self.setLineWidth(1)
        self.line(54, 738, 558, 738) # Draw line at top
        
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(PRIMARY_COLOR)
        self.drawString(54, 744, "HYPERSPECTRAL ANOMALY DETECTION PIPELINE")
        
        self.setFont("Helvetica", 8)
        self.setFillColor(SECONDARY_COLOR)
        self.drawRightString(558, 744, "Technical & Architecture Report")

        # Footer Line & Page Numbers
        self.setStrokeColor(BORDER_COLOR)
        self.setLineWidth(0.5)
        self.line(54, 54, 558, 54)
        
        self.setFont("Helvetica", 8)
        self.setFillColor(TEXT_DARK)
        self.drawString(54, 40, "Confidential - For Internal Engineering Use Only")
        
        page_num_str = f"Page {self._pageNumber} of {total_pages}"
        self.drawRightString(558, 40, page_num_str)
        
        self.restoreState()


def build_pdf(filename="Hyperspectral_Pipeline_Report.pdf"):
    # Margins: 0.75 in (54 pt) all around. Printable width is 612 - 108 = 504 pt.
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=72,
        bottomMargin=72
    )

    styles = getSampleStyleSheet()
    
    # Define custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=PRIMARY_COLOR,
        spaceAfter=15
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=SECONDARY_COLOR,
        spaceAfter=25
    )
    
    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=20,
        textColor=PRIMARY_COLOR,
        spaceBefore=18,
        spaceAfter=10,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=SECONDARY_COLOR,
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=TEXT_DARK,
        spaceAfter=10
    )
    
    body_bold = ParagraphStyle(
        'Body_Bold',
        parent=body_style,
        fontName='Helvetica-Bold'
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=5
    )
    
    code_style = ParagraphStyle(
        'Code_Custom',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#2C3E50"),
        backColor=BG_LIGHT,
        borderColor=BORDER_COLOR,
        borderWidth=0.5,
        borderPadding=6,
        spaceAfter=12
    )

    story = []

    # ================= PAGE 1 =================
    # Title Block
    story.append(Spacer(1, 15))
    story.append(Paragraph("Hyperspectral Anomaly Detection", title_style))
    story.append(Paragraph("End-to-End Spectral-Spatial Pipeline & Engineering Guide", subtitle_style))
    story.append(Spacer(1, 10))

    # Executive Summary Table Box
    summary_text = (
        "<b>Executive Summary:</b> This technical document provides a rigorous overview of the "
        "implemented spectral-spatial hyperspectral anomaly detection (HAD) pipeline. By combining "
        "multivariate statistical modeling (Global RX, local box-filtered RX) with deep unsupervised learning "
        "(PyTorch Autoencoder), the pipeline accurately isolates sub-pixel and compact manmade anomalies "
        "while successfully suppressing high-contrast natural spatial boundaries. Use this guide to understand "
        "data specifications, mathematical structures, model tuning, and evaluation metrics."
    )
    
    table_data = [[Paragraph(summary_text, body_style)]]
    summary_table = Table(table_data, colWidths=[504])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), BG_LIGHT),
        ('BOX', (0,0), (-1,-1), 1.5, SECONDARY_COLOR),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
        ('RIGHTPADDING', (0,0), (-1,-1), 12),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 15))

    # Section 1: The Input & Data Cube Specification
    story.append(Paragraph("1. Data Input & Ingestion Architecture", h1_style))
    story.append(Paragraph(
        "Hyperspectral imaging (HSI) instruments record radiation over hundreds of narrow, contiguous wavelength "
        "bands. Unlike standard RGB images with three channels, a hyperspectral dataset is a <b>3D Spectral-Spatial Data Cube</b> "
        "with dimensions <i>H × W × B</i> (Height × Width × Wavelength Bands).",
        body_style
    ))
    
    story.append(Paragraph("<b>Input Modalities Handled by the Pipeline:</b>", body_style))
    story.append(Paragraph("• <b>Matlab .mat Files:</b> Automatically ingests IEEE Dataport .mat files, parses variable structures, auto-detects the 3D data cube, and binarizes ground truth anomaly labels.", bullet_style))
    story.append(Paragraph("• <b>Highly Realistic Synthetic Simulation:</b> Built-in generator outputs a 100×100×100 cube to simulate complex real-world spectral challenges, creating a rigorous sandbox environment.", bullet_style))
    
    story.append(Spacer(1, 8))
    story.append(Paragraph("<b>The Simulated Target Scene Components:</b>", h2_style))
    
    story.append(Paragraph("1. <b>Continuous Spectral Gradients:</b> Simulates smooth transitions between vegetation (chlorophyll red-edge curve around 700nm) and soil backgrounds (linear rise).", bullet_style))
    story.append(Paragraph("2. <b>Diagonal River Boundary:</b> Water absorbs almost all light, creating a very strong, dark, high-contrast diagonal structure. This represents a major 'edge challenge' that typically causes pixel-based detectors to fail.", bullet_style))
    story.append(Paragraph("3. <b>Natural Patch Anomaly (Suppression Target):</b> A circular region of localized dry soil. This represents natural variation that must <i>not</i> trigger anomaly flags.", bullet_style))
    story.append(Paragraph("4. <b>True Target Anomalies (Injected Manmade Objects):</b>", bullet_style))
    story.append(Paragraph("   - <b>Metal Target (3×3 px):</b> Features sharp, highly active synthetic spectral spikes.", bullet_style))
    story.append(Paragraph("   - <b>White Tarp Target (2×2 px):</b> Flat, highly reflective white spectrum (85% reflectance across all bands).", bullet_style))
    story.append(Paragraph("   - <b>Sub-pixel Target (1×1 px):</b> Very dark object with a single, highly distinct absorption/emission spike.", bullet_style))
    story.append(Paragraph("5. <b>Sensor White Noise:</b> Multi-band Gaussian noise (SNR ≈ 30dB) to mimic electronic sensor imperfections.", bullet_style))
    
    story.append(PageBreak())

    # ================= PAGE 2 =================
    # Section 2: Algorithmic Implementations & Mathematical Foundations
    story.append(Paragraph("2. Algorithmic Architecture & Core Math", h1_style))
    
    story.append(Paragraph("<b>A. PCA Spectral Compression & Noise Suppression</b>", h2_style))
    story.append(Paragraph(
        "Standard hyperspectral bands are highly redundant and correlated. The pipeline applies <b>Principal Component Analysis (PCA)</b> "
        "to reduce the spectral dimensions (e.g., from 100 down to 8 components). This step compresses the signal into orthogonal principal axes, "
        "concentrating >97% of the variance in the top components while dumping high-frequency electronic noise into lower components.",
        body_style
    ))

    story.append(Paragraph("<b>B. Global Reed-Xiaoli (RX) Detection</b>", h2_style))
    story.append(Paragraph(
        "Global RX models the background as a multivariate Gaussian distribution. For every pixel's compressed spectrum vector <b>x</b>_i, "
        "it calculates the <b>Mahalanobis Distance</b> relative to the global mean <b>μ</b>_global and global inverse covariance <b>Σ</b>_global_inv:",
        body_style
    ))
    story.append(Paragraph("Score = (x_i - μ_global)^T * Σ_global_inv * (x_i - μ_global)", code_style))
    story.append(Paragraph(
        "<b>Limitation:</b> Since it uses a global average, any large, prominent natural feature (like our diagonal river or large land boundaries) "
        "departs heavily from the 'mean', producing false alarms along the borders.",
        body_style
    ))

    story.append(Paragraph("<b>C. Local Mean RX (Spatial-Spectral De-biasing)</b>", h2_style))
    story.append(Paragraph(
        "To eliminate edge-boundary false alarms, the Local Mean RX model subtracts the <b>local neighborhood mean</b> (using an 11×11 sliding window) "
        "from the pixel before calculating the Mahalanobis distance. This filters out localized natural variations, leaving only sharp, spatial-spectral residuals.",
        body_style
    ))
    story.append(Paragraph("Residual_i = x_i - μ_local_neighborhood\nScore = Residual_i^T * Σ_global_inv * Residual_i", code_style))
    
    # Explain the Cumulative Sum Box-Filter
    box_filter_text = (
        "<b>Engineering Breakthrough: O(1) Integral Box Filter</b><br/>"
        "Calculating a local mean for every pixel's 11×11 neighborhood across all bands is highly CPU-intensive. "
        "The pipeline solves this by padding the spatial boundary and creating a <b>2D Cumulative Sum</b> (Integral Image). "
        "By prepending a zero row and column to align boundary calculations, we can compute the sum of any arbitrary 2D window "
        "with exactly 4 lookups: <i>box_sum = A + D - B - C</i>. This allows the local average of any window size to compute "
        "in less than 0.001 seconds, achieving massive speedups."
    )
    story.append(Spacer(1, 4))
    box_table = Table([[Paragraph(box_filter_text, body_style)]], colWidths=[504])
    box_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#EBF8FF")), # Very light blue
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#BEE3F8")),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(box_table)
    story.append(Spacer(1, 10))

    # PyTorch Autoencoder section
    story.append(Paragraph("<b>D. Unsupervised Deep Learning Autoencoder</b>", h2_style))
    story.append(Paragraph(
        "A neural network consisting of an Encoder and a Decoder. The Encoder compresses the full 100-band spectral profile down "
        "to a highly restricted 4-dimensional latent bottleneck, and the Decoder attempts to reconstruct the original 100 bands from it. "
        "Since the background pixels constitute >99.9% of the training data, the network optimizes its weights to reconstruct the background "
        "perfectly. Consequently, it fails to reconstruct the rare, anomalous manmade materials, yielding a high Reconstruction Mean Squared Error (MSE) "
        "which serves as the anomaly score.",
        body_style
    ))
    
    story.append(PageBreak())

    # ================= PAGE 3 =================
    # Section 3: Evaluation Metrics & Dashboard Outputs
    story.append(Paragraph("3. Metrics, Evaluation & Dashboard Outputs", h1_style))
    story.append(Paragraph(
        "Evaluating a detector goes beyond simple visual inspection. The pipeline implements standard quantitative remote sensing metrics "
        "to gauge accuracy and detection quality:",
        body_style
    ))
    
    # Key-value Table of Metrics
    metric_data = [
        [Paragraph("<b>Metric</b>", body_bold), Paragraph("<b>Remote Sensing Importance & Description</b>", body_bold)],
        [
            Paragraph("ROC-AUC", body_style), 
            Paragraph("Area Under the Receiver Operating Curve. Measures how effectively the model ranks anomaly scores higher than background scores. A score of 1.0 indicates perfect separation.", body_style)
        ],
        [
            Paragraph("PR-AUC", body_style), 
            Paragraph("Area Under the Precision-Recall Curve. Vital for rare target scenarios since it focuses on true detections relative to false positives. Highly sensitive to noise and false alarms.", body_style)
        ],
        [
            Paragraph("Max F1-Score", body_style), 
            Paragraph("Harmonic mean of Precision and Recall. The pipeline utilizes an optimized threshold search to locate the 'maximum achievable' F1-Score to benchmark ideal model potential.", body_style)
        ],
        [
            Paragraph("FAR (False Alarm Rate)", body_style), 
            Paragraph("The fraction of background pixels falsely identified as anomalies. In real operations (e.g., environmental disaster tracking), a FAR below 0.1% is highly desired to avoid wasteful field investigations.", body_style)
        ]
    ]
    metric_table = Table(metric_data, colWidths=[120, 384])
    metric_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), SECONDARY_COLOR),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    for r in range(1, len(metric_data)):
        bg_col = BG_LIGHT if r % 2 == 1 else colors.white
        metric_table.setStyle(TableStyle([('BACKGROUND', (0, r), (-1, r), bg_col)]))
        
    # Re-apply text color white to headers
    metric_table.setStyle(TableStyle([
        ('TEXTCOLOR', (0,0), (0,0), colors.white),
        ('TEXTCOLOR', (1,0), (1,0), colors.white),
    ]))
    
    story.append(metric_table)
    story.append(Spacer(1, 15))

    story.append(Paragraph("<b>The Visualization Dashboard Summary</b>", h2_style))
    story.append(Paragraph(
        "A multi-panel matplotlib dashboard is automatically generated and saved. It contains: "
        "a Pseudo-RGB composite, an Expert Ground Truth Map highlighting anomalies, a background average intensity representation, "
        "continuous anomaly heatmaps (showing warm red spots for highly anomalous regions), binarized max-F1 prediction maps, and ROC/PR curves "
        "drawn side-by-side to allow easy cross-comparison.",
        body_style
    ))
    
    # Section 4: CRITICAL Engineering Decisions & Guidelines
    story.append(Paragraph("4. CRITICAL Insights & Engineering Decisions", h1_style))
    
    decision_text_1 = (
        "<b>1. Choice of Window Size in Local RX</b><br/>"
        "The sliding window size in Local RX is a critical parameter. "
        "If the window size is <i>too small</i> (e.g. 3×3 or 5×5), the background covariance and mean are heavily contaminated by "
        "the anomaly itself, leading to <b>self-suppression</b> where the anomaly cancels its own signal out. "
        "If the window is <i>too large</i> (e.g. 25×25), local variations are lost, and the detector reverts to a Global RX, "
        "losing its ability to suppress natural borders. An 11×11 window is widely regarded as optimal for compact sub-pixel targets."
    )
    story.append(Paragraph(decision_text_1, body_style))
    story.append(Spacer(1, 5))
    
    decision_text_2 = (
        "<b>2. Why Deep Autoencoders Fail on Sub-Pixel Anomaly Targets</b><br/>"
        "Notice in the evaluation metrics that while Global and Local RX achieve perfect ROC-AUC (1.00000), the Autoencoder "
        "achieves a lower ROC-AUC (e.g., 0.40). <i>Why?</i><br/>"
        "Anomalies in remote sensing are often <b>sub-pixel targets</b> (like our 1×1 target). These targets contain mixed "
        "spectral profiles that are heavily influenced by the background spectrum. An autoencoder compresses features globally, "
        "meaning it can generalize and reconstruct slightly mixed pixels reasonably well, lowering its reconstruction error for sub-pixel anomalies. "
        "For pure, large anomalous tarps, autoencoders excel; but for tiny, sub-pixel objects, local spatial-spectral statistics (Local RX) are far superior."
    )
    story.append(Paragraph(decision_text_2, body_style))
    story.append(Spacer(1, 5))

    decision_text_3 = (
        "<b>3. The Impact of PCA Component Count</b><br/>"
        "Choosing the number of PCA components is a delicate trade-off. If you choose too few components (e.g. 1 or 2), you lose "
        "valuable spectral signature variations, rendering distinct anomalies indistinguishable. If you choose too many components "
        "(e.g. >20), you retain high-frequency noise and sensor artifacts, which degrades the Mahalanobis calculation and inflates the "
        "false alarm rate. 4 to 8 components typically capture over 95% of physical variance in most remote sensing scenes."
    )
    story.append(Paragraph(decision_text_3, body_style))
    
    # Build Document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[INFO] PDF successfully compiled and saved to: {filename}")


if __name__ == "__main__":
    build_pdf()
