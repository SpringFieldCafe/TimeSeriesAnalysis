"""原生 Word 复制、填充与版面核验；只操作 report/ 和 qa/ 内的副本。"""
from pathlib import Path
import json
import hashlib
import shutil
import subprocess
import argparse
import win32com.client

ROOT=Path(__file__).resolve().parent
QA=ROOT/'qa'
REPORT=ROOT/'report'
POPPLER=Path.home()/'.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin/pdftoppm.exe'
WORD_CONTEXT={}


def snapshot(doc):
    last=0
    for i in range(1,doc.Paragraphs.Count+1):
        if doc.Paragraphs.Item(i).Range.Text.strip('\r\x07 \t'):
            last=i
    section=doc.Sections.Item(1).PageSetup
    return dict(last_original_paragraph=last,
        prefix_text=doc.Range(0,doc.Paragraphs.Item(last).Range.End).Text,
        paragraphs=[dict(text=doc.Paragraphs.Item(i).Range.Text,
            style=str(doc.Paragraphs.Item(i).Style),
            alignment=doc.Paragraphs.Item(i).Format.Alignment,
            line_spacing=doc.Paragraphs.Item(i).Format.LineSpacing,
            line_rule=doc.Paragraphs.Item(i).Format.LineSpacingRule,
            space_before=doc.Paragraphs.Item(i).Format.SpaceBefore,
            space_after=doc.Paragraphs.Item(i).Format.SpaceAfter,
            font_size=doc.Paragraphs.Item(i).Range.Font.Size)
            for i in range(1,last+1)],
        page={k:getattr(section,k) for k in ['PageWidth','PageHeight','TopMargin','BottomMargin',
                                            'LeftMargin','RightMargin','HeaderDistance','FooterDistance']},
        source_tables=[doc.Tables.Item(i).Range.WordOpenXML for i in range(1,4)],
        header_footer_signature=header_footer_signature(doc),
        shapes=[dict(type=doc.Shapes.Item(i).Type,width=doc.Shapes.Item(i).Width,
                     height=doc.Shapes.Item(i).Height,left=doc.Shapes.Item(i).Left,
                     top=doc.Shapes.Item(i).Top) for i in range(1,doc.Shapes.Count+1)])


def app_open(path,readonly=False):
    try:
        app=win32com.client.GetActiveObject('Word.Application')
        owned=False
    except Exception:
        app=win32com.client.DispatchEx('Word.Application')
        app.Visible=False
        owned=True
    WORD_CONTEXT[id(app)]=(owned,app.DisplayAlerts)
    app.DisplayAlerts=0
    doc=app.Documents.Open(str(path.resolve()),ConfirmConversions=False,ReadOnly=readonly,
                           AddToRecentFiles=False,Visible=False)
    return app,doc


def close_word(app,doc):
    doc.Close(0)
    owned,alerts=WORD_CONTEXT.pop(id(app))
    if owned:app.Quit(0)
    else:app.DisplayAlerts=alerts


def render(pdf,out):
    out.mkdir(exist_ok=True)
    assert out.resolve().is_relative_to(QA.resolve())
    for p in out.glob('page-*.png'):p.unlink()
    subprocess.run([str(POPPLER),'-r','125','-png',str(pdf),str(out/'page')],check=True)


def prepare_template():
    source=next(p for p in ROOT.glob('*.doc') if not p.name.startswith('~$'))
    app,doc=app_open(ROOT/'backup'/source.name,True)
    try:
        snap=snapshot(doc)
        (QA/'template_snapshot.json').write_text(json.dumps(snap,ensure_ascii=False,indent=2),encoding='utf-8')
        questions=[doc.Paragraphs.Item(i).Range.Text.rstrip('\r') for i in range(43,51)]
        (QA/'questions.json').write_text(json.dumps(questions,ensure_ascii=False,indent=2),encoding='utf-8')
        doc.ExportAsFixedFormat(str((QA/'reference.pdf').resolve()),17)
        # 转换仅用于内部结构/渲染检查；交付仍保留.doc格式。
        doc.SaveAs2(str((QA/'reference.docx').resolve()),FileFormat=16,AddToRecentFiles=False)
    finally:
        close_word(app,doc)
    render(QA/'reference.pdf',QA/'reference_pages')
    (QA/'artifact.md').write_text(
        '# 模板填充约定\n\n原件保持字节不变，使用report/副本；保留原始前50段及3个表格、2个图形对象。\n'
        '页面几何、段落及表格证据见template_snapshot.json，参考页图见reference_pages/。\n'
        '仅从最后一个题目之后填充：删除副本尾部110个空段，新增分页后逐题复制题干并解答。\n'
        '新增文字沿用宋体10.5磅；图宽不超过12.6厘米；只为新增表格和文字设直接格式。\n'
        '禁止全局改样式、修改原始题干、页眉页脚、页面尺寸、封面或评分栏；可自然增加页数。\n'
        '原件校验见source_hashes.json；所有最终页面须逐页目视检查。\n',encoding='utf-8')
    print('Template reference prepared',flush=True)


def add_text(doc,text,kind='p'):
    start=doc.Content.End-1
    r=doc.Range(start,start);r.InsertAfter(text+'\r')
    r=doc.Range(start,start+len(text)+1)
    r.Font.Name='Times New Roman';r.Font.NameFarEast='宋体';r.Font.Size=10.5
    r.Font.Bold=kind in ['heading','question','title']
    r.Font.Color=0
    fmt=r.ParagraphFormat
    fmt.Alignment=1 if kind in ['caption','title'] else 3
    fmt.FirstLineIndent=21 if kind=='p' else 0
    fmt.SpaceBefore=5 if kind in ['heading','question'] else 0
    fmt.SpaceAfter=4 if kind!='code' else 0
    fmt.LineSpacingRule=1;fmt.KeepTogether=True
    fmt.KeepWithNext=kind in ['heading','question','reference']
    if kind=='code':
        r.Font.Name='Consolas';r.Font.Size=8.5;fmt.Alignment=0;fmt.LineSpacingRule=0
    if kind=='caption':r.Font.Size=9
    if kind in ['reference','reference_last']:r.Font.Size=9.5
    if kind=='title':r.Font.Size=12
    return r


def add_table(doc,block):
    rows=[block['headers'],*block['rows']]
    r=doc.Range(doc.Content.End-1,doc.Content.End-1)
    table=doc.Tables.Add(r,len(rows),len(rows[0]))
    table.AllowAutoFit=False
    table.Rows.LeftIndent=5.4
    table.Range.Font.Name='Times New Roman';table.Range.Font.NameFarEast='宋体';table.Range.Font.Size=9
    table.Range.ParagraphFormat.SpaceAfter=2;table.Range.ParagraphFormat.SpaceBefore=2
    table.Range.ParagraphFormat.LineSpacingRule=0
    table.Range.ParagraphFormat.KeepWithNext=False
    table.Borders.Enable=True
    for row_i,values in enumerate(rows,1):
        for col_i,value in enumerate(values,1):
            cell=table.Cell(row_i,col_i);cell.Range.Text=str(value)
            cell.VerticalAlignment=1
            cell.Range.ParagraphFormat.Alignment=0 if col_i==1 else 1
            if row_i==1:
                cell.Range.Font.Bold=True;cell.Shading.BackgroundPatternColor=15132390
        table.Rows.Item(row_i).AllowBreakAcrossPages=False
        table.Rows.Item(row_i).Range.ParagraphFormat.KeepWithNext=row_i<len(rows)
    table.Rows.Item(1).HeadingFormat=True
    widths=block.get('widths',[12.6/len(rows[0])]*len(rows[0]))
    for col_i,width in enumerate(widths,1):table.Columns.Item(col_i).Width=width*28.3464567
    # 表格后的专用段落可容纳后续图文。
    r=doc.Range(table.Range.End,table.Range.End);r.InsertParagraphAfter()


def build():
    blocks=json.loads((QA/'report_blocks.json').read_text(encoding='utf-8'))
    source=next(p for p in ROOT.glob('*.doc') if not p.name.startswith('~$'))
    working=REPORT/'course_report_working.doc'
    final=REPORT/'时间序列分析课程考查报告_完成副本.doc'
    shutil.copy2(ROOT/'backup'/source.name,working)
    app,doc=app_open(working)
    try:
        baseline=json.loads((QA/'template_snapshot.json').read_text(encoding='utf-8'))
        end=doc.Paragraphs.Item(baseline['last_original_paragraph']).Range.End
        doc.Range(end,doc.Content.End-1).Delete()
        r=doc.Range(doc.Content.End-1,doc.Content.End-1);r.InsertBreak(7)
        for block in blocks:
            kind=block['kind']
            if kind=='image':
                r=doc.Range(doc.Content.End-1,doc.Content.End-1)
                pic=doc.InlineShapes.AddPicture(str((ROOT/block['path']).resolve()),False,True,r)
                pic.LockAspectRatio=True;pic.Width=block.get('width',12.4)*28.3464567
                pic.Range.ParagraphFormat.Alignment=1
                pic.Range.ParagraphFormat.KeepWithNext=True
                doc.Range(pic.Range.End,pic.Range.End).InsertParagraphAfter()
            elif kind=='table':add_table(doc,block)
            else:add_text(doc,block['text'],kind)
        doc.Repaginate()
        for item in baseline['header_footer_signature']:
            collection=doc.Sections.Item(1).Headers if item['kind']=='header' else doc.Sections.Item(1).Footers
            collection.Item(item['index']).Range.Fields.Update()
        doc.SaveAs2(str(final.resolve()),FileFormat=0,AddToRecentFiles=False)
        doc.ExportAsFixedFormat(str((QA/'final.pdf').resolve()),17)
        doc.SaveAs2(str((QA/'final.docx').resolve()),FileFormat=16,AddToRecentFiles=False)
        after=snapshot_prefix(doc,baseline['last_original_paragraph'])
        checks={k:after[k]==baseline[k] for k in ['prefix_text','paragraphs','page','shapes']}
        if 'header_footer_signature' in baseline:
            checks['header_footer_preserved']=header_footer_signature(doc)==baseline['header_footer_signature']
        checks['source_tables_text_preserved']=all(
            text_only_xml(doc.Tables.Item(i).Range.WordOpenXML)==text_only_xml(baseline['source_tables'][i-1])
            for i in range(1,4))
        checks['all_source_hashes_unchanged']=all(
            hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==v['sha256']
            for name,v in json.loads((QA/'source_hashes.json').read_text(encoding='utf-8')).items())
        checks['page_count']=doc.ComputeStatistics(2)
        checks['inline_images']=doc.InlineShapes.Count
        (QA/'template_verification.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
        assert all(v for k,v in checks.items() if isinstance(v,bool)),checks
    finally:
        close_word(app,doc)
    render(QA/'final.pdf',QA/'final_pages')
    print('Report built:',final,checks,flush=True)


def text_only_xml(xml):
    import xml.etree.ElementTree as ET
    return [e.text for e in ET.fromstring(xml).iter() if e.tag.endswith('}t')]


def header_footer_signature(doc):
    result=[]
    for s in doc.Sections:
        for label,collection in [('header',s.Headers),('footer',s.Footers)]:
            indices=[1]
            if s.PageSetup.DifferentFirstPageHeaderFooter:indices.append(2)
            if s.PageSetup.OddAndEvenPagesHeaderFooter:indices.append(3)
            for i in indices:
                h=collection.Item(i)
                if not h.Exists:continue
                r=h.Range;text=r.Text
                for j in range(r.Fields.Count,0,-1):
                    f=r.Fields.Item(j)
                    # Word的Range偏移含域代码，Text不含；不可直接把两者当同一坐标。
                    value=f.Result.Text;start=text.rfind(value)
                    if value and start>=0:
                        text=text[:start]+'FIELD['+f.Code.Text.strip()+']'+text[start+len(value):]
                result.append(dict(kind=label,index=i,text=text))
    return result


def snapshot_prefix(doc,last):
    # 快照仅比较原始前缀，不将新增段落列入模板一致性判断。
    section=doc.Sections.Item(1).PageSetup
    return dict(prefix_text=doc.Range(0,doc.Paragraphs.Item(last).Range.End).Text,
        paragraphs=[dict(text=doc.Paragraphs.Item(i).Range.Text,
            style=str(doc.Paragraphs.Item(i).Style),alignment=doc.Paragraphs.Item(i).Format.Alignment,
            line_spacing=doc.Paragraphs.Item(i).Format.LineSpacing,
            line_rule=doc.Paragraphs.Item(i).Format.LineSpacingRule,
            space_before=doc.Paragraphs.Item(i).Format.SpaceBefore,
            space_after=doc.Paragraphs.Item(i).Format.SpaceAfter,
            font_size=doc.Paragraphs.Item(i).Range.Font.Size) for i in range(1,last+1)],
        page={k:getattr(section,k) for k in ['PageWidth','PageHeight','TopMargin','BottomMargin',
                                           'LeftMargin','RightMargin','HeaderDistance','FooterDistance']},
        shapes=[dict(type=doc.Shapes.Item(i).Type,width=doc.Shapes.Item(i).Width,
                     height=doc.Shapes.Item(i).Height,left=doc.Shapes.Item(i).Left,
                     top=doc.Shapes.Item(i).Top) for i in range(1,doc.Shapes.Count+1)])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','build']);a=p.parse_args()
    prepare_template() if a.action=='prepare' else build()
