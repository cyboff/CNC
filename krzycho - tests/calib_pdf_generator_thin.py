# generate_calib_pdf.py
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

MM2IN = 1/25.4
PT_PER_MM = 72/25.4

PAGE_MM   = 70.0
SQUARE_MM = 40.0
LINE_W_MM = 0.10     # mm (tenká čára)
MARKER_MM = 9.0      # mm (vnější strana značky)
CLEAR_MM  = 2.0      # mm (mezera od rámečku)
MARGIN_MM = (PAGE_MM - SQUARE_MM) / 2

aruco = cv2.aruco
DICT  = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

def make_marker(id_, px=800, border_bits=2):
    # borderBits=2 → větší tiché pole = spolehlivější detekce
    img = aruco.generateImageMarker(DICT, id_, px, borderBits=border_bits)
    return img

IDS = {"TL":0, "TR":1, "BR":2, "BL":3}
markers = {k: make_marker(v, px=1000, border_bits=2) for k, v in IDS.items()}

fig = plt.figure(figsize=(PAGE_MM*MM2IN, PAGE_MM*MM2IN))
ax = plt.axes([0,0,1,1]); ax.set_xlim(0,PAGE_MM); ax.set_ylim(0,PAGE_MM)
ax.set_aspect('equal'); ax.axis('off')

# 40×40 mm tenký rámeček
ax.add_patch(Rectangle((MARGIN_MM, MARGIN_MM), SQUARE_MM, SQUARE_MM,
                       fill=False, linewidth=LINE_W_MM*PT_PER_MM, edgecolor='black'))

# křížky v rozích
cross_len = 3.0
def cross(x,y, hx, vy):
    ax.plot([x, x+hx*cross_len],[y,y], linewidth=LINE_W_MM*PT_PER_MM, color='black')
    ax.plot([x,x],[y,y+vy*cross_len], linewidth=LINE_W_MM*PT_PER_MM, color='black')
cross(MARGIN_MM,                 MARGIN_MM+SQUARE_MM,  1,-1)  # TL
cross(MARGIN_MM+SQUARE_MM,      MARGIN_MM+SQUARE_MM, -1,-1)  # TR
cross(MARGIN_MM+SQUARE_MM,      MARGIN_MM,           -1, 1)  # BR
cross(MARGIN_MM,                 MARGIN_MM,            1, 1)  # BL

# pozice markerů (uvnitř čtverce s mezerou CLEAR_MM)
corners = {
    "TL": (MARGIN_MM + CLEAR_MM,                         MARGIN_MM + SQUARE_MM - CLEAR_MM),
    "TR": (MARGIN_MM + SQUARE_MM - CLEAR_MM - MARKER_MM, MARGIN_MM + SQUARE_MM - CLEAR_MM),
    "BR": (MARGIN_MM + SQUARE_MM - CLEAR_MM - MARKER_MM, MARGIN_MM + CLEAR_MM + MARKER_MM),
    "BL": (MARGIN_MM + CLEAR_MM,                         MARGIN_MM + CLEAR_MM + MARKER_MM),
}
for key, img in markers.items():
    x0, y_anchor = corners[key]
    if key in ("TL","TR"):
        extent = [x0, x0+MARKER_MM, y_anchor-MARKER_MM, y_anchor]
    else:
        y0 = y_anchor - MARKER_MM
        extent = [x0, x0+MARKER_MM, y0, y0+MARKER_MM]
    ax.imshow(img, cmap='gray', interpolation='nearest', extent=extent, origin='lower')

# orientace + měřítko
ax.text(PAGE_MM/2, PAGE_MM-2, "UP", fontsize=8, ha='center', va='top')
bar_len=30.0; bar_x0=(PAGE_MM-bar_len)/2; bar_y=MARGIN_MM*0.5
ax.plot([bar_x0, bar_x0+bar_len], [bar_y, bar_y], linewidth=LINE_W_MM*PT_PER_MM, color='black')
ax.text(bar_x0, bar_y+1.5, "30.00 mm scale", fontsize=8, ha='left', va='bottom')

fig.savefig("calibration_40mm_aruco_b2_clear2.pdf", format="pdf")
fig.savefig("calibration_40mm_aruco_b2_clear2.png", dpi=300)
print("Saved calibration_40mm_aruco_b2_clear2.pdf")
