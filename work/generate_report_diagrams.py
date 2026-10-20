from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

OUT = Path(r"C:\Users\Guman Behera\Documents\Codex\2026-10-20\files-pasted-by-the-user-project\outputs")
BLUE = "#1565C0"; BLACK = "#111111"; WHITE = "#FFFFFF"; PALE = "#F7FBFF"

def font(size=24, bold=False):
    name = "arialbd.ttf" if bold else "arial.ttf"
    return ImageFont.truetype(Path("C:/Windows/Fonts") / name, size)

def canvas(title, size=(1600, 900)):
    im = Image.new("RGB", size, WHITE); d = ImageDraw.Draw(im)
    d.text((70, 45), title, font=font(36, True), fill=BLACK)
    d.line((70, 105, size[0]-70, 105), fill=BLUE, width=3)
    return im, d

def box(d, xy, label, size=23):
    x1,y1,x2,y2 = xy
    d.rounded_rectangle(xy, radius=16, fill=PALE, outline=BLUE, width=4)
    lines = label.split("\n")
    heights = [d.textbbox((0,0), line, font=font(size, True))[3] for line in lines]
    total = sum(heights) + 8*(len(lines)-1)
    y = (y1+y2-total)/2
    for line,h in zip(lines,heights):
        w = d.textbbox((0,0), line, font=font(size, True))[2]
        d.text(((x1+x2-w)/2,y), line, font=font(size, True), fill=BLACK)
        y += h+8

def arrow(d, start, end):
    d.line((*start,*end), fill=BLUE, width=4)
    import math
    angle = math.atan2(end[1]-start[1], end[0]-start[0])
    for offset in (math.pi - 0.5, math.pi + 0.5):
        point=(end[0]+20*math.cos(angle+offset),end[1]+20*math.sin(angle+offset))
        d.line((*end,*point), fill=BLUE, width=4)

def architecture():
    im,d=canvas("System Architecture")
    labels=["User","React\nFrontend","FastAPI\nBackend","Text\nPreprocessing","Tokenizer &\nPadding","BiLSTM\nModel","Class\nProbabilities","Results UI"]
    xs=[70,240,440,650,860,1080,1280,1440]
    widths=[120,150,160,170,170,150,160,120]
    for x,w,label in zip(xs,widths,labels): box(d,(x,315,x+w,465),label,20)
    for i in range(len(xs)-1): arrow(d,(xs[i]+widths[i],390),(xs[i+1],390))
    box(d,(635,640,870,775),"SQLite History",24); arrow(d,(520,465),(710,640)); arrow(d,(710,640),(710,465))
    d.text((70,830),"Blue outlines represent application components; arrows represent data flow.",font=font(19),fill=BLACK)
    im.save(OUT/"system-architecture.png")

def pipeline():
    im,d=canvas("Model Training Pipeline")
    labels=["Fact-checked\nDataset","Cleaning &\nValidation","Stratified\nSplit","Train-only\nTokenizer","Sequence\nPadding","BiLSTM\nTraining","Evaluation","Saved Model\nand Metrics"]
    coords=[(85,250),(285,250),(485,250),(685,250),(885,250),(1085,250),(1285,250),(1285,540)]
    for (x,y),label in zip(coords,labels): box(d,(x,y,x+170,y+135),label,19)
    for i in range(6): arrow(d,(coords[i][0]+170,318),(coords[i+1][0],318))
    arrow(d,(1370,385),(1370,540))
    d.text((85,770),"The tokenizer is fitted only on training text to prevent data leakage.",font=font(22),fill=BLACK)
    im.save(OUT/"training-pipeline.png")

def flow():
    im,d=canvas("Three-Class Classification Flow")
    steps=[("Input Text",(620,150,980,245)),("Normalize Unicode\nand Whitespace",(620,300,980,415)),("Tokenize and\nPad to 300 Tokens",(620,470,980,585)),("BiLSTM Softmax\nOutput",(620,640,980,755))]
    for label,xy in steps: box(d,xy,label,25)
    for i in range(3): arrow(d,((steps[i][1][0]+steps[i][1][2])//2,steps[i][1][3]),((steps[i+1][1][0]+steps[i+1][1][2])//2,steps[i+1][1][1]))
    outputs=[("Likely Real",(140,785,450,875)),("Needs Verification",(645,785,955,875)),("Likely Fake",(1150,785,1460,875))]
    for label,xy in outputs: box(d,xy,label,22)
    arrow(d,(700,755),(295,785)); arrow(d,(800,755),(800,785)); arrow(d,(900,755),(1305,785))
    im.save(OUT/"classification-flow.png")

def database():
    im,d=canvas("SQLite Database Design")
    d.rounded_rectangle((430,175,1170,700), radius=16, fill=PALE, outline=BLUE, width=4)
    d.text((710,205),"ANALYSES",font=font(30,True),fill=BLACK)
    rows=["id — Integer, Primary Key","preview — Truncated text preview","prediction — Model output label","confidence — Highest class probability","created_at — Analysis timestamp"]
    y=310
    for row in rows:
        d.line((480,y-16,1120,y-16),fill=BLUE,width=2)
        d.text((510,y),row,font=font(24),fill=BLACK); y+=70
    d.line((480,y-16,1120,y-16),fill=BLUE,width=2)
    d.text((430,775),"Only a short text preview is stored locally; full input text is not logged.",font=font(22),fill=BLACK)
    im.save(OUT/"database-design.png")

def confusion_format():
    im,d=canvas("Confusion Matrix Format", (1600, 850))
    actual = ["Actual Likely Real", "Actual Needs Verification", "Actual Likely Fake"]
    predicted = "Predicted: Real / Verify / Fake"
    ys = [190, 405, 620]
    for label, y in zip(actual, ys):
        box(d, (180, y, 660, y + 125), label, 24)
        box(d, (930, y, 1450, y + 125), predicted, 23)
        arrow(d, (660, y + 62), (930, y + 62))
    d.text((180, 790), "Rows represent actual classes; columns in the measured matrix represent predicted classes.", font=font(20), fill=BLACK)
    im.save(OUT/"confusion-matrix-format.png")

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    architecture(); pipeline(); flow(); database(); confusion_format()
    print(f"Created PNG diagrams in {OUT}")
