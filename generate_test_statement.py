from fpdf import FPDF
from datetime import date, timedelta
import random

transactions = [
    ("AMAZON PURCHASE", -49.99), ("SALARY DEPOSIT", 3500.00),
    ("NETFLIX", -15.99), ("WHOLE FOODS", -87.43),
    ("UBER TRIP", -12.50), ("ATM WITHDRAWAL", -200.00),
    ("STARBUCKS", -6.75), ("RENT PAYMENT", -1500.00),
    ("ELECTRIC BILL", -94.20), ("GYM MEMBERSHIP", -39.99),
    ("AMAZON PURCHASE", -49.99),  # intentional duplicate for anomaly detection
]

pdf = FPDF()
pdf.add_page()
pdf.set_font("Helvetica", "B", 16)
pdf.cell(0, 10, "First National Bank - Statement", ln=True, align="C")
pdf.set_font("Helvetica", size=10)
pdf.cell(0, 8, "Account: 1234567890  |  John Doe  |  SSN: 123-45-6789", ln=True)
pdf.ln(5)

pdf.set_font("Helvetica", "B", 10)
pdf.cell(35, 8, "Date");pdf.cell(90, 8, "Description")
pdf.cell(35, 8, "Amount"); pdf.cell(30, 8, "Balance", ln=True)
pdf.set_font("Helvetica", size=10)

balance = 5000.00
start   = date(2024, 1, 1)

for i, (desc, amount) in enumerate(transactions):
    txn_date = start + timedelta(days=i * 2)
    balance += amount
    pdf.cell(35, 7, str(txn_date))
    pdf.cell(90, 7, desc)
    pdf.cell(35, 7, f"${amount:,.2f}")
    pdf.cell(30, 7, f"${balance:,.2f}", ln=True)

pdf.output("test_statement.pdf")
print("Generated test_statement.pdf")