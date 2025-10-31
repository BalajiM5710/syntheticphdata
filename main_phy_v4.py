import os
import json
import random
import string
import qrcode
import cv2
import numpy as np
from tqdm import tqdm
from faker import Faker
import glob

from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.lib.enums import TA_CENTER, TA_RIGHT

from PIL import Image as PILImage, ImageEnhance, ImageFilter

from pdf2image import convert_from_path

NUM_SAMPLES = 300
OUTPUT_DIR = "durable_invoices_dataset_v4"
POPPLER_PATH = r"C:\Users\Dell\Downloads\Release-25.07.0-0\poppler-25.07.0\Library\bin" 

PAPER_TEXTURE_PATHS = glob.glob("assets/paper*.jpg")
if not PAPER_TEXTURE_PATHS:
    print("Warning: No paper textures found in 'assets/paper*.jpg'. Background will be white.")
else:
    print(f"Found {len(PAPER_TEXTURE_PATHS)} paper textures to use.")

fake = Faker('en_IN')

VALID_HSN_CODES = {
    "Smartphones": "8517", "Laptops": "8471", "Tablets": "8471",
    "Television": "8528", "Refrigerator": "8418", "Washing Machine": "8450",
    "Air Conditioner": "8415", "Microwave": "8516", "Camera": "8525",
    "Smartwatch": "8517", "Headphones": "8518", "Speaker": "8518", "Bike": "8711"
}

INDIAN_BRANDS = {
    "Smartphones": ["Samsung", "OnePlus", "Xiaomi", "Realme", "Vivo", "Oppo", "Apple", "Motorola"],
    "Laptops": ["HP", "Dell", "Lenovo", "Asus", "Acer", "Apple"],
    "Tablets": ["Samsung", "Apple", "Lenovo", "Xiaomi"],
    "Television": ["Samsung", "LG", "Sony", "Mi", "OnePlus", "TCL"],
    "Refrigerator": ["LG", "Samsung", "Whirlpool", "Godrej", "Haier"],
    "Washing Machine": ["LG", "Samsung", "Whirlpool", "IFB", "Bosch"],
    "Air Conditioner": ["Voltas", "LG", "Samsung", "Daikin", "Blue Star"],
    "Microwave": ["LG", "Samsung", "IFB", "Bajaj", "Whirlpool"],
    "Camera": ["Canon", "Nikon", "Sony", "Fujifilm"],
    "Smartwatch": ["Apple", "Samsung", "Noise", "Fire-Boltt", "Amazfit"],
    "Headphones": ["Sony", "JBL", "Boat", "Sennheiser"],
    "Speaker": ["JBL", "Sony", "Boat", "Bose"],
    "Bike": ["TVS","Royal Enfield", "Hero", "Bajaj"]
}

DEALER_NAMES = [
    "ELECTRONICS WORLD", "MOBILE POINT", "TECH BAZAAR", "DIGITAL STORE",
    "SMART ELECTRONICS", "FUTURE TECH", "GADGET GALAXY", "SUPREME ELECTRONICS",
    "CITY ELECTRONICS", "ROYAL MOBILES", "STAR ELECTRONICS", "METRO TECH STORE",
    "VISHWA ELECTRONICS", "PRADHAN MOBILE SHOP", "RAM ELECTRONICS",
    "KUMAR TECH SOLUTIONS", "SINGH ELECTRONICS", "PATEL MOBILES"
]


def generate_gstin():
    state_code = f"{random.randint(1, 37):02d}"; pan = "".join(random.choices(string.ascii_uppercase, k=5)) + "".join(random.choices(string.digits, k=4)) + random.choice(string.ascii_uppercase)
    return f"{state_code}{pan}1Z{random.choice('0123456789ABCDEFGHJKLMNPQRSTUVWXYZ')}"

def generate_long_hex(): return ''.join(random.choices('0123456789abcdef', k=64))
def generate_imei(): return "".join([str(random.randint(0, 9)) for _ in range(15)])

def num_to_words(num):
    to_19 = 'Zero One Two Three Four Five Six Seven Eight Nine Ten Eleven Twelve Thirteen Fourteen Fifteen Sixteen Seventeen Eighteen Nineteen'.split()
    tens = 'Twenty Thirty Forty Fifty Sixty Seventy Eighty Ninety'.split()
    def words(n):
        if n < 20: return to_19[n]
        if n < 100: return tens[n//10-2] + (" " + to_19[n%10] if n%10 else "")
        if n < 1000: return to_19[n//100] + " Hundred" + (" and " + words(n%100) if n%100 else "")
        if n < 100000: return words(n//1000) + " Thousand" + (" " + words(n%1000) if n%1000 else "")
        if n < 10000000: return words(n//100000) + " Lakh" + (" " + words(n%100000) if n%100000 else "")
        return "Many"
    return words(int(num)).title()


def generate_invoice_data():
    
    line_items_for_pdf = []
    items_for_gt_parse = []
    subtotal, qty_total = 0, 0
    tax_rate = 0.18

    for i in range(random.randint(1, 4)):
        category = random.choice(list(INDIAN_BRANDS.keys()))
        brand = random.choice(INDIAN_BRANDS[category])
        model = f"{brand[:3].upper()}-{random.randint(100, 999)}{random.choice(string.ascii_uppercase)}"
        hsn = VALID_HSN_CODES[category]
        qty = random.randint(1, 2)
        rate = round(random.uniform(8000.0, 95000.0), 2)
        if category == "Bike": rate = round(random.uniform(80000.0, 250000.0), 2)
        elif category in ["Laptops", "Television", "Air Conditioner"]: rate = round(random.uniform(30000.0, 150000.0), 2)
        
        amount = qty * rate
        subtotal += amount
        qty_total += qty
        imei = generate_imei() if category == "Smartphones" else None
        
        
        item_desc = f"{brand} {category} {model}"
        if imei: item_desc += f" IMEI: {imei}"
        
        items_for_gt_parse.append({
            "item_desc": item_desc,
            "item_qty": str(qty),
            "item_net_price": f"{rate:,.2f}",
            "item_net_worth": f"{amount:,.2f}",
            "item_vat": f"{int(tax_rate * 100)}%",
            "item_gross_worth": f"{amount * (1 + tax_rate):,.2f}"
        })

        
        desc_for_pdf = f"<b>{brand} {category}</b><br/>Model: {model}"
        if imei: desc_for_pdf += f"<br/>IMEI: {imei}"
        desc_for_dotmatrix = f"{brand} {category} Model:{model}"
        if imei: desc_for_dotmatrix += f" IMEI:{imei}"

        line_items_for_pdf.append({
            "sl_no": i + 1, "description": desc_for_pdf, "description_dm": desc_for_dotmatrix,
            "hsn": hsn, "qty": str(qty), "rate": f"₹{rate:,.2f}", "amount": f"₹{amount:,.2f}"
        })

    tax_amount = subtotal * tax_rate
    total = subtotal + tax_amount

    
    pdf_data = {
        "invoice_no": f"{''.join(random.choices(string.ascii_uppercase, k=2))}-{random.randint(1000,9999)}",
        "invoice_date": fake.date_this_year().strftime('%d/%m/%Y'),
        "dealer_name": random.choice(DEALER_NAMES), "dealer_gstin": generate_gstin(),
        "dealer_address": fake.address().replace('\n', ', '),
        "customer_name": fake.name().upper(), "customer_address": fake.address().replace('\n', ', '),
        "customer_phone": fake.phone_number(), "customer_gstin": generate_gstin(), "customer_state": fake.state(),
        "total_value": f"₹{total:,.2f}", "line_items": line_items_for_pdf, "qty_total": str(qty_total),
        "subtotal_value": f"₹{subtotal:,.2f}", "tax_rate_percent": f"{int(tax_rate*100)}%",
        "tax_amount_value": f"₹{tax_amount:,.2f}", "total_in_words": f"Indian Rupee {num_to_words(total)} Only",
        "irn": generate_long_hex(), "ack_no": str(random.randint(10**9, 10**10 - 1)),
        "ack_date": fake.date_this_year().strftime('%d-%b-%y'),
        "dealer_name_at_top": random.choice([True, False])
    }
    
    
    ground_truth = {
        "gt_parse": {
            "header": {
                "invoice_number": pdf_data['invoice_no'],
                "invoice_date": pdf_data['invoice_date'],
                "seller": f"{pdf_data['dealer_name']} {pdf_data['dealer_address']}",
                "client": f"{pdf_data['customer_name']} {pdf_data['customer_address']}",
                "seller_tax_id": pdf_data['dealer_gstin'],
                "client_tax_id": pdf_data['customer_gstin'],
                "client_phone": pdf_data['customer_phone']
            },
            "items": items_for_gt_parse,
            "summary": {
                "total_net_worth": pdf_data['subtotal_value'],
                "total_vat": pdf_data['tax_amount_value'],
                "total_gross_worth": pdf_data['total_value']
            }
        }
    }

    return pdf_data, ground_truth


def create_layout_tally_style(data, pdf_path):
    doc = SimpleDocTemplate(pdf_path, pagesize=letter, leftMargin=30, rightMargin=30, topMargin=30, bottomMargin=30)
    story, styles = [], getSampleStyleSheet()
    styles.add(ParagraphStyle(name='small', fontSize=7))
    styles.add(ParagraphStyle(name='bold', fontName='Helvetica-Bold'))
    styles.add(ParagraphStyle(name='CenterH1', fontSize=14, fontName='Helvetica-Bold', alignment=TA_CENTER))
    right_aligned_style = ParagraphStyle(name='RightAlign', parent=styles['Normal'], alignment=TA_RIGHT)

    if data['dealer_name_at_top']:
        story.append(Paragraph(data['dealer_name'], styles['CenterH1']))
        story.append(Spacer(1, 0.2*inch))

    qr_img_path = f"qr_temp_{data['invoice_no']}.png"; qrcode.make(data['irn']).save(qr_img_path); qr_img = Image(qr_img_path, width=1*inch, height=1*inch)
    header_data = [[Paragraph("<b>Tax Invoice</b>", styles['h1']), Paragraph("e-Invoice", styles['Normal']), qr_img]]; header_table = Table(header_data, colWidths=[5.5*inch, 1*inch, 1*inch]); header_table.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'TOP'), ('ALIGN', (0,0), (0,0), 'CENTER')])); story.append(header_table)
    irn_block = Paragraph(f"<b>IRN:</b> <font size=7>{data['irn']}</font><br/><b>Ack. No. :</b> {data['ack_no']}<br/><b>Ack. Date :</b> {data['ack_date']}", styles['Normal']); story.append(irn_block); story.append(Spacer(1, 0.2*inch))
    
    vendor_details = Paragraph(f"""<b>{data['dealer_name']}</b><br/>{data['dealer_address']}<br/><b>GSTIN/UIN:</b> {data['dealer_gstin']}<br/><b>State Name:</b> {data['customer_state']}""", styles['Normal'])
    buyer_details = Paragraph(f"""<b>Buyer (Bill to)</b><br/><b>{data['customer_name']}</b><br/>{data['customer_address']}<br/><b>Ph:</b> {data['customer_phone']}<br/><b>GSTIN/UIN:</b> {data['customer_gstin']}<br/><b>State Name:</b> {data['customer_state']}""", styles['Normal'])
    
    invoice_details_data = [[Paragraph("Invoice No.", styles['small']), Paragraph("Dated", styles['small'])],[Paragraph(f"<b>{data['invoice_no']}</b>", styles['Normal']), Paragraph(f"<b>{data['invoice_date']}</b>", styles['Normal'])]]
    invoice_details_table = Table(invoice_details_data, colWidths=[1.7*inch, 1.7*inch]); invoice_details_table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.5, colors.black)]))
    
    main_details_data = [[vendor_details, invoice_details_table], [buyer_details, ""]]; main_details_table = Table(main_details_data, colWidths=[3.8*inch, 3.7*inch]); main_details_table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.5, colors.black), ('SPAN', (1,0), (1,1)), ('VALIGN', (0,0), (-1,-1), 'TOP')])); story.append(main_details_table)
    
    item_header = [Paragraph(h, styles['small']) for h in ["Sl No.", "Description of Goods", "HSN/SAC", "Quantity", "Rate", "Amount"]]; item_data = [item_header]
    for item in data['line_items']:
        item_data.append([
            Paragraph(str(item['sl_no'])), Paragraph(item['description'], styles['Normal']),
            Paragraph(item['hsn']), Paragraph(item['qty']),
            Paragraph(item['rate'], right_aligned_style), Paragraph(item['amount'], right_aligned_style)
        ])
    for _ in range(5 - len(data['line_items'])): item_data.append([""] * 6)
    
    total_amount_para = Paragraph(f"<b>{data['total_value']}</b>", right_aligned_style)
    item_data.append([Paragraph("Total", styles['bold']), "", "", Paragraph(data['qty_total'], styles['Normal']), "", total_amount_para])
    
    item_table = Table(item_data, colWidths=[0.4*inch, 4*inch, 0.8*inch, 0.8*inch, 1*inch, 1.2*inch]); item_table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.5, colors.black), ('VALIGN', (0,0), (-1,-1), 'MIDDLE'), ('ALIGN', (1,1), (1,-2), 'LEFT'), ('ALIGN', (0,0), (0,-1), 'CENTER'), ('SPAN', (1,-1), (2,-1))])); story.append(item_table)
    story.append(Paragraph(f"Amount Chargeable (in words)<br/><b>{data['total_in_words']}</b>", styles['Normal']))
    
    tax_summary_data = [
        ["", "Subtotal", data['subtotal_value']],
        ["", f"IGST @ {data['tax_rate_percent']}", data['tax_amount_value']],
        ["", Paragraph("<b>TOTAL</b>", styles['bold']), Paragraph(f"<b>{data['total_value']}</b>", styles['bold'])]
    ]
    tax_table = Table(tax_summary_data, colWidths=[5.3*inch, 1.2*inch, 1.2*inch]); tax_table.setStyle(TableStyle([('GRID', (1,0), (-1,-1), 0.5, colors.black), ('ALIGN', (1,0), (-1,-1), 'RIGHT')])); story.append(tax_table)
    doc.build(story); os.remove(qr_img_path)

def create_layout_landscape(data, pdf_path):
    doc = SimpleDocTemplate(pdf_path, pagesize=landscape(letter), leftMargin=0.5*inch, rightMargin=0.5*inch, topMargin=0.5*inch, bottomMargin=0.5*inch)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='bold', fontName='Helvetica-Bold'))
    styles.add(ParagraphStyle(name='right', alignment=TA_RIGHT))
    styles.add(ParagraphStyle(name='CenterH1', fontSize=16, fontName='Helvetica-Bold', alignment=TA_CENTER, spaceBottom=20))
    story = []

    if data['dealer_name_at_top']:
        story.append(Paragraph(data['dealer_name'], styles['CenterH1']))
    else:
        story.append(Paragraph("<b>TAX INVOICE</b>", styles['h1']))
        
    story.append(Spacer(1, 0.2*inch))
    
    details_data = [
        [Paragraph(f"<b>Sold By:</b><br/>{data['dealer_name']}<br/>{data['dealer_address']}<br/>GSTIN: {data['dealer_gstin']}", styles['Normal']),
         Paragraph(f"<b>Billed To:</b><br/>{data['customer_name']}<br/>{data['customer_address']}<br/>Ph: {data['customer_phone']}<br/>GSTIN: {data['customer_gstin']}", styles['Normal']),
         Paragraph(f"Invoice No: <b>{data['invoice_no']}</b><br/>Date: <b>{data['invoice_date']}</b>", styles['Normal'])]
    ]
    details_table = Table(details_data, colWidths=[3.5*inch, 3.5*inch, 3*inch]); story.append(details_table); story.append(Spacer(1, 0.2*inch))
    
    item_header = [Paragraph(h, styles['bold']) for h in ["#", "Asset Description", "HSN", "Qty", "Rate", "Amount"]]; item_data = [item_header]
    for item in data['line_items']: item_data.append([Paragraph(str(item['sl_no'])), Paragraph(item['description']), Paragraph(item['hsn']), Paragraph(item['qty']), Paragraph(item['rate'], styles['right']), Paragraph(item['amount'], styles['right'])])
    
    item_table = Table(item_data, colWidths=[0.5*inch, 5*inch, 1*inch, 1*inch, 1.2*inch, 1.3*inch]); item_table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.5, colors.black), ('BACKGROUND', (0,0), (-1,0), colors.lightgrey), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')])); story.append(item_table)
    
    summary_data = [
        ['Subtotal', data['subtotal_value']],
        [f"IGST @ {data['tax_rate_percent']}", data['tax_amount_value']],
        [Paragraph("<b>Total</b>", styles['bold']), Paragraph(f"<b>{data['total_value']}</b>", styles['bold'])]
    ]
    summary_table = Table(summary_data, colWidths=[1.2*inch, 1.3*inch], hAlign='RIGHT'); summary_table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.5, colors.black), ('ALIGN', (0,0), (-1,-1), 'RIGHT')])); story.append(summary_table)
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph(f"<b>In Words:</b> {data['total_in_words']}", styles['Normal']))
    doc.build(story)

def create_layout_dotmatrix(data, pdf_path):
    c = canvas.Canvas(pdf_path, pagesize=letter); width, height = letter
    c.setFont("Courier", 9)
    y = height - 0.5*inch
    
    if data['dealer_name_at_top']:
        c.setFont("Courier-Bold", 12)
        c.drawCentredString(width/2, y, data['dealer_name'])
        y -= 0.2*inch
        c.setFont("Courier", 9)
        c.drawCentredString(width/2, y, "TAX INVOICE")
    else:
        c.drawCentredString(width/2, y, "TAX INVOICE")

    y -= 0.2*inch; c.line(0.5*inch, y, width - 0.5*inch, y); y -= 0.15*inch
    c.drawString(0.5*inch, y, f"SOLD BY: {data['dealer_name']}"); c.drawRightString(width-0.5*inch, y, f"INVOICE #: {data['invoice_no']}")
    y -= 0.15*inch; c.drawString(0.5*inch, y, data['dealer_address'][:60]); c.drawRightString(width-0.5*inch, y, f"DATE: {data['invoice_date']}")
    y -= 0.15*inch; c.drawString(0.5*inch, y, f"GSTIN: {data['dealer_gstin']}")
    y -= 0.3*inch; c.drawString(0.5*inch, y, f"BILLED TO: {data['customer_name']}")
    y -= 0.15*inch; c.drawString(0.5*inch, y, f"ADDRESS: {data['customer_address'][:60]}")
    y -= 0.15*inch; c.drawString(0.5*inch, y, f"PHONE: {data['customer_phone']}")
    y -= 0.15*inch; c.drawString(0.5*inch, y, f"GSTIN: {data['customer_gstin']}")
    y -= 0.2*inch; c.line(0.5*inch, y + 0.1*inch, width - 0.5*inch, y + 0.1*inch)
    c.drawString(0.6*inch, y, "ITEM DESCRIPTION"); c.drawString(4.0*inch, y, "HSN"); c.drawString(5.0*inch, y, "QTY"); c.drawString(6.0*inch, y, "RATE"); c.drawString(7.2*inch, y, "AMOUNT")
    c.line(0.5*inch, y - 0.1*inch, width - 0.5*inch, y - 0.1*inch)
    y -= 0.25*inch
    
    for item in data['line_items']:
        desc_parts = item['description_dm'].split(" IMEI:")
        c.drawString(0.6*inch, y, desc_parts[0]);
        c.drawRightString(4.5*inch, y, item['hsn']); c.drawRightString(5.5*inch, y, item['qty']);
        c.drawRightString(6.8*inch, y, item['rate']); c.drawRightString(8.0*inch, y, item['amount']);
        if len(desc_parts) > 1: y -= 0.15*inch; c.drawString(0.8*inch, y, f"IMEI:{desc_parts[1]}")
        y -= 0.2*inch

    y -= 0.1*inch; c.line(5.8*inch, y, width - 0.5*inch, y); y -= 0.2*inch
    c.drawRightString(6.8*inch, y, "Subtotal:"); c.drawRightString(8.0*inch, y, data['subtotal_value'])
    y -= 0.2*inch; c.drawRightString(6.8*inch, y, f"IGST @ {data['tax_rate_percent']}:"); c.drawRightString(8.0*inch, y, data['tax_amount_value'])
    y -= 0.2*inch; c.setFont("Courier-Bold", 10); c.drawRightString(6.8*inch, y, "TOTAL:"); c.drawRightString(8.0*inch, y, data['total_value']); c.save()


def generate_invoice_image_final(base_filename, poppler_path=None):
    pdf_path = f"{base_filename}.pdf"
    pdf_data, ground_truth_data = generate_invoice_data()
    layouts = [create_layout_tally_style, create_layout_landscape, create_layout_dotmatrix]
    chosen_layout = random.choice(layouts)
    try: chosen_layout(pdf_data, pdf_path)
    except Exception as e: print(f"Error generating PDF with {chosen_layout.__name__}: {e}"); return None, None
    images = convert_from_path(pdf_path, dpi=random.randint(200, 250), poppler_path=poppler_path)
    if os.path.exists(pdf_path): os.remove(pdf_path)
    if images: return images[0], ground_truth_data
    return None, None




def apply_realistic_photo_effects_physical(image):
    paper_texture_cv = None
    if PAPER_TEXTURE_PATHS: paper_texture_cv = cv2.imread(random.choice(PAPER_TEXTURE_PATHS), cv2.IMREAD_GRAYSCALE)

    def add_paper_texture(img, texture):
        if texture is None: return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rows, cols = img.shape[:2]; resized_texture = cv2.resize(texture, (cols, rows))
        if len(img.shape) == 3: img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else: img_gray = img
        inv_img = cv2.bitwise_not(img_gray)
        blended = cv2.multiply(resized_texture.astype(float), cv2.bitwise_not(inv_img).astype(float), scale=1/255)
        return blended.astype(np.uint8)

    def add_fold_crease(img):
        rows, cols = img.shape[:2]; crease_overlay = np.zeros_like(img, dtype=np.uint8)
        crease_x = int(cols * random.uniform(0.45, 0.55)); crease_width = random.randint(15, 25)
        cv2.line(crease_overlay, (crease_x, 0), (crease_x, rows), (20, 20, 20), thickness=crease_width)
        crease_overlay = cv2.GaussianBlur(crease_overlay, (31, 31), 0)
        return cv2.subtract(img, crease_overlay)

    def add_uneven_lighting(img):
        rows, cols = img.shape[:2]
        kernel_x = cv2.getGaussianKernel(cols, int(cols*random.uniform(0.4,0.7))); kernel_y = cv2.getGaussianKernel(rows, int(rows*random.uniform(0.4,0.7)))
        kernel = kernel_y * kernel_x.T; mask = kernel / np.max(kernel)
        min_b, max_b = random.uniform(0.6, 0.75), random.uniform(1.1, 1.3); scaled_mask = min_b + (mask * (max_b - min_b))
        lighting_mask = cv2.merge([scaled_mask,scaled_mask,scaled_mask]).astype(np.float32)
        float_image = img.astype(np.float32) / 255.0
        multiplied = cv2.multiply(float_image, lighting_mask)
        return np.clip(multiplied * 255, 0, 255).astype(np.uint8)

    def add_stamp_overlay(img_pil):
        if random.random() < 0.4: return img_pil
        try: stamp = PILImage.open("assets/stamp.png").convert("RGBA")
        except: return img_pil
        base_w, base_h = img_pil.size; scale = random.uniform(0.15, 0.25)
        new_w = int(base_w*scale); new_h = int((stamp.height/stamp.width)*new_w)
        stamp = stamp.resize((new_w, new_h), PILImage.Resampling.LANCZOS)
        stamp = stamp.rotate(random.randint(-25, 25), expand=True, resample=PILImage.Resampling.BICUBIC)
        start_x, end_x = int(base_w * 0.6), int(base_w * 0.95 - stamp.width)
        px = start_x if start_x >= end_x else random.randint(start_x, end_x)
        start_y, end_y = int(base_h * 0.7), int(base_h * 0.95 - stamp.height)
        py = start_y if start_y >= end_y else random.randint(start_y, end_y)
        img_pil.paste(stamp, (px, py), stamp)
        return img_pil

    cv_image = np.array(image.convert('RGB'))[:, :, ::-1]
    degraded = cv2.GaussianBlur(cv_image, (3,3), random.uniform(0.3, 0.6))
    textured = add_paper_texture(degraded, paper_texture_cv)
    textured_bgr = cv2.cvtColor(textured, cv2.COLOR_GRAY2BGR)
    rows, cols = textured_bgr.shape[:2]
    pts1 = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])
    x_m, y_m = cols*0.05, rows*0.03
    dx1, dy1, dx2, dy2 = random.uniform(-x_m, x_m), random.uniform(-y_m, y_m), random.uniform(-x_m, x_m), random.uniform(-y_m, y_m)
    pts2 = np.float32([[dx1, dy1], [cols - dx2, dy2], [0, rows], [cols, rows]])
    M_persp = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(textured_bgr, M_persp, (cols, rows), borderValue=(230, 230, 230))
    angle = random.uniform(-2.5, 2.5)
    M_rot = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    rotated = cv2.warpAffine(warped, M_rot, (cols, rows), borderValue=(230, 230, 230))
    creased = add_fold_crease(rotated)
    lit = add_uneven_lighting(creased)
    final_pil = PILImage.fromarray(cv2.cvtColor(lit, cv2.COLOR_BGR2RGB))
    final_pil = add_stamp_overlay(final_pil)
    return final_pil




def main():
    img_dir = os.path.join(OUTPUT_DIR, "images")
    ann_dir = os.path.join(OUTPUT_DIR, "annotations")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)
    print(f"Generating {NUM_SAMPLES} definitive, consumer durable invoices...")

    for i in tqdm(range(NUM_SAMPLES)):
        base_filename = f"invoice_{i:05d}"
        clean_image, ground_truth_data = generate_invoice_image_final(base_filename, poppler_path=POPPLER_PATH)
        if clean_image is None: continue
        
        realistic_image = apply_realistic_photo_effects_physical(clean_image)
        
        final_image_path = os.path.join(img_dir, f"{base_filename}.jpg")
        realistic_image.save(final_image_path, "JPEG", quality=random.randint(65, 90))
        annotation_path = os.path.join(ann_dir, f"{base_filename}.json")
        with open(annotation_path, 'w', encoding='utf-8') as f:
            json.dump(ground_truth_data, f, indent=2, ensure_ascii=False)
    print(f"\nDefinitive dataset generated in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()