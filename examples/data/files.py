def secret_pdf() -> bytes:
    return b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 144] /Contents 4 0 R >>
endobj
4 0 obj
<< /Length 44 >>
stream
BT
/F1 24 Tf
100 100 Td
(secret: pineapple) Tj
ET
endstream
endobj
xref
0 5
0000000000 65535 f 
0000000010 00000 n 
0000000061 00000 n 
0000000114 00000 n 
0000000173 00000 n 
trailer
<< /Root 1 0 R /Size 5 >>
startxref
243
%%EOF
"""
