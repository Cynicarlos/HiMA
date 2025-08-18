# app.py
from flask import Flask, render_template, request, send_file
import os
from PIL import Image
import uuid
import time
from threading import Thread
# 确保使用绝对路径
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'static/uploads')
app.config['PROCESSED_FOLDER'] = os.path.join(BASE_DIR, 'static/processed')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)


def process_image(input_path, output_path):
    """更健壮的处理函数"""
    try:
        with Image.open(input_path) as img:
            # 保留原始格式
            file_format = img.format
            gray_img = img.convert('L')
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 使用原始格式保存
            gray_img.save(output_path, format=file_format)
            print(f"[Debug] 图片已保存到: {output_path}")
    except Exception as e:
        print(f"[Error] 处理过程中出错: {str(e)}")
        raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return {'status': 'error'}
    
    file = request.files['file']
    if file.filename == '':
        return {'status': 'error'}
    
    filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)
    return {'status': 'success', 'filename': filename}

@app.route('/process', methods=['POST'])
def process():
    input_filename = request.json['filename']
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
    
    # 添加路径验证
    if not os.path.exists(input_path):
        return {'error': '原始文件不存在'}, 400

    output_filename = 'processed_' + input_filename
    output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
    
    # 添加处理日志
    print(f"[Debug] 输入路径: {input_path}")
    print(f"[Debug] 输出路径: {output_path}")
    
    try:
        process_image(input_path, output_path)
        print("[Debug] 文件处理完成")
    except Exception as e:
        print(f"[Error] 处理失败: {str(e)}")
        return {'error': '图片处理失败'}, 500

    # 验证输出文件
    if not os.path.exists(output_path):
        print("[Error] 输出文件未创建")
        return {'error': '处理结果未生成'}, 500
        
    return {'processed_filename': output_filename}

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(
        os.path.join(app.config['PROCESSED_FOLDER'], filename),
        as_attachment=True
    )

if __name__ == '__main__':
    app.run(debug=True)