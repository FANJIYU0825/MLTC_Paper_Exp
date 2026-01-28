"""
PDF to Markdown Converter
將 PDF 每一頁轉換成 Markdown 章節
"""
import os

def pdf_to_markdown(pdf_path, output_path):
    """
    將 PDF 轉換為 Markdown 格式
    每一頁對應一個 ## 章節
    """
    try:
        import PyPDF2
    except ImportError:
        print("正在安裝 PyPDF2...")
        os.system("pip install pypdf2 --quiet")
        import PyPDF2
    
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        total_pages = len(reader.pages)
        
        # 取得 PDF 檔名作為標題
        pdf_name = os.path.basename(pdf_path).replace('.pdf', '')
        
        markdown_content = f"# {pdf_name}\n\n"
        markdown_content += f"**來源檔案**: `{pdf_path}`\n"
        markdown_content += f"**總頁數**: {total_pages} 頁\n\n"
        markdown_content += "---\n\n"
        
        for i, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            
            # 清理文字
            if text:
                # 移除多餘空白
                text = text.strip()
                # 保留換行但移除過多的空行
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                cleaned_text = '\n\n'.join(lines)
            else:
                cleaned_text = "(此頁無法提取文字)"
            
            markdown_content += f"## 第 {i} 頁\n\n"
            markdown_content += f"{cleaned_text}\n\n"
            markdown_content += "---\n\n"
        
        # 寫入檔案
        with open(output_path, 'w', encoding='utf-8') as out:
            out.write(markdown_content)
        
        print(f"✅ 轉換完成！")
        print(f"   - 輸入: {pdf_path}")
        print(f"   - 輸出: {output_path}")
        print(f"   - 頁數: {total_pages}")
        
        return markdown_content

if __name__ == "__main__":
    # PDF 路徑
    pdf_path = ".agent/.skills/report_skills/references/投影片筆記.pdf"
    # 輸出路徑
    output_path = ".agent/.skills/report_skills/references/投影片筆記.md"
    
    content = pdf_to_markdown(pdf_path, output_path)
    
    # 顯示部分內容預覽
    print("\n" + "="*50)
    print("內容預覽 (前 3000 字元):")
    print("="*50)
    print(content[:3000])
