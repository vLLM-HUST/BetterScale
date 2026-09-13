"""Render the canonical Markdown report to a self-contained, printable HTML file.

Requires Markdown==3.8.2 in a documentation-only environment, not the donor venv.
The serving package has no dependency on this renderer.
"""
from pathlib import Path
import re
import markdown

root=Path(__file__).resolve().parent
source=(root/'REPORT.zh-CN.md').read_text()
def inline_svg(match):
    name=match.group(2)
    path=(root/name).resolve()
    if path.parent!=root/'figures' or path.suffix!='.svg':
        raise ValueError('Only local report SVG assets may be embedded')
    svg=path.read_text().replace('<svg ','<svg role="img" ',1)
    return '\n<div class="figure">'+svg+'</div>\n'
source=re.sub(r'!\[([^\]]*)\]\((figures/[^)]+\.svg)\)',inline_svg,source)
body=markdown.markdown(source,extensions=['tables','fenced_code','toc'])
style='''
:root{color-scheme:light;font-family:system-ui,"Noto Sans CJK SC","Microsoft YaHei",sans-serif;color:#172b42;background:#edf2f7}
body{max-width:1120px;margin:32px auto;padding:48px 60px;background:white;box-shadow:0 12px 45px #15263814;line-height:1.85}
h1{font-size:34px;line-height:1.4;letter-spacing:-.025em;border-bottom:4px solid #328b76;padding-bottom:24px}
h2{font-size:25px;margin-top:46px;color:#206f60}h3{font-size:20px;margin-top:30px}
p,li{font-size:16px}strong{color:#143c52}a{color:#237ba5;text-underline-offset:3px}
code{font-family:ui-monospace,monospace;background:#edf3f6;border-radius:4px;padding:2px 5px;font-size:.9em}
pre{overflow-x:auto;background:#102a3d;color:#e8f1f6;border-radius:10px;padding:20px 24px;line-height:1.6}
pre code{background:none;color:inherit;padding:0}table{border-collapse:collapse;width:100%;margin:24px 0;font-size:14px}
th,td{border-bottom:1px solid #d8e3eb;padding:12px 14px;text-align:left;vertical-align:top}th{background:#edf5f2}
.figure{margin:30px -28px}.figure svg{display:block;width:100%;height:auto}hr{border:0;border-top:1px solid #d8e3eb;margin:36px 0}
@media(max-width:800px){body{margin:0;padding:24px}.figure{margin:24px -12px}h1{font-size:27px}table{font-size:12px}}
@media print{body{max-width:none;margin:0;padding:0;box-shadow:none}h2,h3{break-after:avoid}table,.figure{break-inside:avoid}.figure{margin:20px 0}pre{white-space:pre-wrap}a{color:inherit}}
'''
(root/'REPORT.zh-CN.html').write_text('<!doctype html>\n<html lang="zh-CN"><head><meta charset="utf-8">'
    '<meta name="viewport" content="width=device-width, initial-scale=1"><title>DSV4 执行路径改造</title>'
    '<style>'+style+'</style></head><body>'+body+'</body></html>\n')
print(root/'REPORT.zh-CN.html')
