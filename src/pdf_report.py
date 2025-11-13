"""
PDF Report Generator for bi-SIS Community Targeting Analysis
"""
import io
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, 
    PageBreak, Image, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def create_pdf_report(
    graph_name,
    graph_stats,
    params,
    critical_params,
    budget_results=None,
    epsilon_results=None,
    figures=None,
    filename="analysis_report.pdf"
):
    """
    Generate a comprehensive PDF report for bi-SIS analysis.
    
    Args:
        graph_name: Name of the graph
        graph_stats: Dict with N, E, avg_degree
        params: Dict with tau1, tau2, cost_scheme, etc.
        critical_params: Dict with mu_c, u_c_stats
        budget_results: DataFrame with budget sweep results
        epsilon_results: DataFrame with epsilon sweep results
        figures: Dict of matplotlib figures
        filename: Output PDF filename
    
    Returns:
        BytesIO object containing the PDF
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, 
                           topMargin=0.75*inch, bottomMargin=0.75*inch)
    
    # Container for the 'Flowable' objects
    elements = []
    
    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=12,
        textColor=colors.HexColor('#34495e'),
        spaceAfter=8,
        fontName='Helvetica-Bold'
    )
    
    normal_style = styles['Normal']
    
    # Title Page
    elements.append(Spacer(1, 0.5*inch))
    elements.append(Paragraph("bi-SIS Community Targeting", title_style))
    elements.append(Paragraph("Analysis Report", title_style))
    elements.append(Spacer(1, 0.3*inch))
    
    # Report metadata
    meta_text = f"""
    <para align=center>
    <b>Graph:</b> {graph_name}<br/>
    <b>Generated:</b> {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}<br/>
    <b>Implementation:</b> Keeping Up with the Winner (arXiv:2403.19903)
    </para>
    """
    elements.append(Paragraph(meta_text, normal_style))
    elements.append(Spacer(1, 0.5*inch))
    
    # Table of Contents
    elements.append(Paragraph("Table of Contents", heading_style))
    toc_items = [
        "1. Executive Summary",
        "2. Graph Information",
        "3. Model Parameters",
        "4. Critical Parameters Analysis",
        "5. Budget Sweep Results",
        "6. Epsilon Sweep Results",
        "7. Method Comparison",
        "8. Visualizations",
        "9. Conclusions"
    ]
    for item in toc_items:
        elements.append(Paragraph(f"&nbsp;&nbsp;&nbsp;&nbsp;{item}", normal_style))
    
    elements.append(PageBreak())
    
    # 1. Executive Summary
    elements.append(Paragraph("1. Executive Summary", heading_style))
    summary_text = f"""
    This report presents a comprehensive analysis of competing product diffusion in the network 
    <b>{graph_name}</b> using the bi-SIS (bidirectional Susceptible-Infected-Susceptible) model. 
    The analysis evaluates the optimal community targeting strategy for a new product (Product 2) 
    competing against an established dominant product (Product 1).
    <br/><br/>
    <b>Key Findings:</b>
    """
    elements.append(Paragraph(summary_text, normal_style))
    elements.append(Spacer(1, 12))
    
    # Add key findings if results available
    if critical_params:
        findings_data = [
            ["Metric", "Value"],
            ["Critical Intervention (μ_c)", f"{critical_params.get('mu_c', 'N/A'):.6g}"],
            ["Network Size (N)", str(graph_stats['N'])],
            ["Average Degree", f"{graph_stats['avg_degree']:.2f}"],
        ]
        
        if budget_results is not None and len(budget_results) > 0:
            max_y = budget_results['AvgY'].max()
            findings_data.append(["Max Market Share (Product 2)", f"{max_y:.4f}"])
        
        findings_table = Table(findings_data, colWidths=[3*inch, 2*inch])
        findings_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),
        ]))
        elements.append(findings_table)
    
    elements.append(PageBreak())
    
    # 2. Graph Information
    elements.append(Paragraph("2. Graph Information", heading_style))
    elements.append(Paragraph("Network Statistics", subheading_style))
    
    graph_data = [
        ["Property", "Value"],
        ["Number of Nodes (N)", str(graph_stats['N'])],
        ["Number of Edges (E)", str(graph_stats['E'])],
        ["Average Degree", f"{graph_stats['avg_degree']:.2f}"],
        ["Graph Type", graph_name],
    ]
    
    graph_table = Table(graph_data, colWidths=[3*inch, 2*inch])
    graph_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2ecc71')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),
    ]))
    elements.append(graph_table)
    elements.append(Spacer(1, 20))
    
    # 3. Model Parameters
    elements.append(Paragraph("3. Model Parameters", heading_style))
    
    param_data = [
        ["Parameter", "Symbol", "Value", "Description"],
        ["Tau 1", "τ₁", f"{params['tau1']:.3f}", "Spreading rate for Product 1"],
        ["Tau 2", "τ₂", f"{params['tau2']:.3f}", "Spreading rate for Product 2"],
        ["Cost Scheme", "w_i", params['cost_scheme'], "Node cost function"],
        ["Epsilon Min", "ε_min", f"{params.get('eps_min', 'N/A'):.1e}", "Minimum perturbation"],
        ["Epsilon Max", "ε_max", f"{params.get('eps_max', 'N/A'):.1e}", "Maximum perturbation"],
        ["Local Search Iterations", "-", str(params.get('iters', 1)), "Optimization iterations"],
    ]
    
    param_table = Table(param_data, colWidths=[1.5*inch, 0.8*inch, 1.2*inch, 2.5*inch])
    param_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e74c3c')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),
    ]))
    elements.append(param_table)
    
    elements.append(PageBreak())
    
    # 4. Critical Parameters Analysis
    elements.append(Paragraph("4. Critical Parameters Analysis", heading_style))
    
    if critical_params:
        crit_text = f"""
        The critical parameters define the minimum intervention needed for Product 2 to survive 
        in competition with Product 1. These are computed using <b>Lemma 3.2</b> from the research paper.
        <br/><br/>
        <b>Critical Intervention Strength (μ_c):</b> {critical_params['mu_c']:.6g}
        <br/>
        This represents the minimum boost needed in the network structure for Product 2.
        """
        elements.append(Paragraph(crit_text, normal_style))
        elements.append(Spacer(1, 12))
        
        if 'u_c_stats' in critical_params:
            elements.append(Paragraph("Critical Targeting Vector (u_c) Statistics:", subheading_style))
            uc_stats = critical_params['u_c_stats']
            
            uc_data = [
                ["Statistic", "Value"],
                ["Mean", f"{uc_stats.get('mean', 0):.6f}"],
                ["Std Dev", f"{uc_stats.get('std', 0):.6f}"],
                ["Min", f"{uc_stats.get('min', 0):.6f}"],
                ["Max", f"{uc_stats.get('max', 0):.6f}"],
                ["Sum (Expected Size)", f"{uc_stats.get('sum', 0):.6f}"],
            ]
            
            uc_table = Table(uc_data, colWidths=[2.5*inch, 2*inch])
            uc_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#9b59b6')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),
            ]))
            elements.append(uc_table)
    
    elements.append(PageBreak())
    
    # 5. Budget Sweep Results
    if budget_results is not None and len(budget_results) > 0:
        elements.append(Paragraph("5. Budget Sweep Results", heading_style))
        
        budget_text = f"""
        The budget sweep analysis shows how Product 2's market share changes with different 
        advertising budgets. A total of <b>{len(budget_results)}</b> budget levels were analyzed.
        """
        elements.append(Paragraph(budget_text, normal_style))
        elements.append(Spacer(1, 12))
        
        # Summary statistics
        elements.append(Paragraph("Budget Sweep Summary:", subheading_style))
        
        summary_data = [
            ["Metric", "Value"],
            ["Budget Range", f"{budget_results['Budget'].min():.4f} - {budget_results['Budget'].max():.4f}"],
            ["Max Market Share (Product 2)", f"{budget_results['AvgY'].max():.6f}"],
            ["Max Market Share (Product 1)", f"{budget_results['AvgX'].max():.6f}"],
            ["Max Community Size", f"{budget_results['ExpectedCommunitySize'].max():.4f}"],
        ]
        
        summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#16a085')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),
        ]))
        elements.append(summary_table)
        elements.append(Spacer(1, 12))
        
        # Top 5 results
        elements.append(Paragraph("Top 5 Budget Configurations:", subheading_style))
        top_budget = budget_results.nlargest(5, 'AvgY')[['Budget', 'AvgX', 'AvgY', 'ExpectedCommunitySize']]
        
        budget_detail_data = [["Budget", "Product 1", "Product 2", "Community Size"]]
        for _, row in top_budget.iterrows():
            budget_detail_data.append([
                f"{row['Budget']:.4f}",
                f"{row['AvgX']:.6f}",
                f"{row['AvgY']:.6f}",
                f"{row['ExpectedCommunitySize']:.4f}"
            ])
        
        budget_detail_table = Table(budget_detail_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
        budget_detail_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),
        ]))
        elements.append(budget_detail_table)
    
    elements.append(PageBreak())
    
    # 6. Epsilon Sweep Results
    if epsilon_results is not None and len(epsilon_results) > 0:
        elements.append(Paragraph("6. Epsilon Sweep Results", heading_style))
        
        methods = epsilon_results['method'].unique()
        eps_text = f"""
        The epsilon sweep optimizes the community targeting strategy using Algorithm 1 
        (local perturbation search). Results are compared across <b>{len(methods)}</b> different methods.
        """
        elements.append(Paragraph(eps_text, normal_style))
        elements.append(Spacer(1, 12))
        
        # Method-wise summary
        for method in methods:
            method_data = epsilon_results[epsilon_results['method'] == method]
            if len(method_data) > 0:
                elements.append(Paragraph(f"Method: {method}", subheading_style))
                
                method_summary = [
                    ["Metric", "Value"],
                    ["Best Product 2 Market Share", f"{method_data['AvgY'].max():.6f}"],
                    ["Best Product 1 Market Share", f"{method_data['AvgX'].max():.6f}"],
                    ["Average Product 2 Share", f"{method_data['AvgY'].mean():.6f}"],
                ]
                
                method_table = Table(method_summary, colWidths=[3*inch, 2*inch])
                method_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#95a5a6')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),
                ]))
                elements.append(method_table)
                elements.append(Spacer(1, 8))
    
    elements.append(PageBreak())
    
    # 7. Method Comparison
    if epsilon_results is not None and len(epsilon_results) > 0:
        elements.append(Paragraph("7. Method Comparison", heading_style))
        
        # Aggregate comparison
        comparison_data = [["Method", "Max Product 2", "Avg Product 2", "Performance"]]
        
        for method in methods:
            method_data = epsilon_results[epsilon_results['method'] == method]
            max_y = method_data['AvgY'].max()
            avg_y = method_data['AvgY'].mean()
            
            # Simple performance rating
            if max_y > 0.1:
                perf = "Excellent"
            elif max_y > 0.05:
                perf = "Good"
            elif max_y > 0.01:
                perf = "Fair"
            else:
                perf = "Poor"
            
            comparison_data.append([method, f"{max_y:.6f}", f"{avg_y:.6f}", perf])
        
        comparison_table = Table(comparison_data, colWidths=[2*inch, 1.5*inch, 1.5*inch, 1.5*inch])
        comparison_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e67e22')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),
        ]))
        elements.append(comparison_table)
    
    elements.append(PageBreak())
    
    # 8. Visualizations
    elements.append(Paragraph("8. Visualizations", heading_style))
    
    if figures:
        for fig_name, fig in figures.items():
            elements.append(Paragraph(fig_name, subheading_style))
            
            # Convert figure to image
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            
            # Add image to PDF
            img = Image(img_buffer, width=6*inch, height=4*inch)
            elements.append(img)
            elements.append(Spacer(1, 12))
            
            plt.close(fig)
    
    elements.append(PageBreak())
    
    # 9. Conclusions
    elements.append(Paragraph("9. Conclusions", heading_style))
    
    conclusion_text = f"""
    This comprehensive analysis of the <b>{graph_name}</b> network using the bi-SIS model 
    demonstrates the effectiveness of strategic community targeting for product diffusion.
    <br/><br/>
    <b>Key Takeaways:</b>
    <br/>
    • The critical intervention parameter μ_c = {critical_params.get('mu_c', 'N/A'):.6g} defines the minimum 
    boost needed for Product 2 to survive against the dominant Product 1.
    <br/>
    """
    
    if budget_results is not None and len(budget_results) > 0:
        conclusion_text += f"""
        • Budget sweep analysis across {len(budget_results)} configurations shows that Product 2 
        can achieve a maximum market share of {budget_results['AvgY'].max():.4f}.
        <br/>
        """
    
    if epsilon_results is not None and len(epsilon_results) > 0:
        best_method = epsilon_results.loc[epsilon_results['AvgY'].idxmax(), 'method']
        best_score = epsilon_results['AvgY'].max()
        conclusion_text += f"""
        • Among the tested methods, <b>{best_method}</b> achieved the best performance with a 
        Product 2 market share of {best_score:.6f}.
        <br/>
        """
    
    conclusion_text += """
    <br/>
    <b>Recommendations:</b>
    <br/>
    • Implement the optimal community targeting strategy identified by Algorithm 1
    <br/>
    • Allocate budget according to the critical threshold and identified sweet spots
    <br/>
    • Monitor market dynamics and adjust intervention as network evolves
    <br/><br/>
    <i>This report was generated using the implementation from "Keeping Up with the Winner! 
    Targeted Advertisement to Communities in Social Networks" (arXiv:2403.19903)</i>
    """
    
    elements.append(Paragraph(conclusion_text, normal_style))
    
    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer
