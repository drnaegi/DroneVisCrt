"""Show the median and lowest paired PSNR differences on the frozen Sand Key flight."""
from PIL import Image, ImageDraw, ImageFont

from relation_nvs.io import load_camera, sha256
from tools.benchmark_stereo_transport import OUT, ROOT, raster, read


def plot():
    folder = OUT/'sand_key'
    scores = read(folder/'scores.json')['rows']
    rows = {(r['case'],r['method']):r for r in scores}
    cases = [r['case'] for r in scores if r['method']=='rife_ground']
    fixed = 'detail_qualified_fixed'
    def diff(case):
        return rows[case,fixed]['psnr'] - rows[case,'rife_ground']['psnr']
    cases.sort(key=diff)
    manifest = {r['case']:r for r in read(folder/'predictions.json')}
    methods = [('Target RGB',None),('Aligned RIFE','rife_ground'),
               ('Median depth + two bands','detail_median_stereo_fixed'),('Qualified depth + two bands',fixed)]
    font = ImageFont.truetype('DejaVuSans.ttf', 15)
    title = ImageFont.truetype('DejaVuSans-Bold.ttf', 19)
    width, block = 272, 307
    fig = Image.new('RGB',(4*width+10,2*block+14),'white')
    label = ImageDraw.Draw(fig)
    for i,(rank,case) in enumerate([('Median',cases[len(cases)//2]),('Lowest',cases[0])]):
        top = 5+i*block
        rec = manifest[case]
        camera = load_camera(rec['cameras'][2])
        target = ROOT/rec['paths'][2]
        if sha256(target) != rows[case,fixed]['target_sha256']:
            raise ValueError(f'Target changed for {case}')
        label.text((7,top),f'{rank} paired PSNR difference    Sand Key {target.stem}    {diff(case):+.2f} dB',font=title,fill='#202124')
        for j,(name,method) in enumerate(methods):
            x = 7 + j*width
            if method is None:
                im = Image.fromarray(raster(target,camera))
            else:
                with Image.open(folder/case/(method+'.png')) as src:
                    im = src.convert('RGB')
            label.text((x,top+30),name,font=font,fill='#202124')
            thumb = im.copy()
            thumb.thumbnail((width-8,145),Image.Resampling.LANCZOS)
            left = x+(width-8-thumb.width)//2
            fig.paste(thumb,(left,top+52))
            # An identical central crop in all methods and both targets; no content-based selection.
            cx,cy = im.width//2,im.height//2
            crop = (cx-64,cy-24,cx+64,cy+24)
            box = [left+int(crop[0]*thumb.width/im.width),top+52+int(crop[1]*thumb.height/im.height),
                   left+int(crop[2]*thumb.width/im.width),top+52+int(crop[3]*thumb.height/im.height)]
            label.rectangle(box,outline='white',width=1)
            fig.paste(im.crop(crop).resize((256,96),Image.Resampling.NEAREST),(x+4,top+203))
    fig.save(folder/'paired_examples.png')


if __name__ == '__main__':
    plot()
