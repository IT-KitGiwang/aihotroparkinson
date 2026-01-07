"""
Hệ thống AI Hỗ trợ Phát hiện Sớm Parkinson
Phiên bản cải tiến với:
- Phân tích tổng hợp (Ảnh + Triệu chứng)
- Tối ưu hóa prompt với cảnh báo y tế
- Không hướng dẫn dùng thuốc
- Khuyến khích thăm khám bác sĩ
"""

import os
import base64
import io
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Flask app
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "default_secret_key")

# Configure Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemma-3-12b-it")

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    gemini_model = genai.GenerativeModel(GEMINI_MODEL)
    print(f"✅ Gemini model initialized: {GEMINI_MODEL}")
else:
    gemini_model = None
    print("⚠️ Warning: GOOGLE_API_KEY not found")

# ========================================
# LOAD PYTORCH MODEL
# ========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "MoHinh", "mo_hinh_AI.pth")

model = None
if os.path.exists(MODEL_PATH):
    try:
        model = models.resnet18()
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model = model.to(DEVICE)
        model.eval()
        print(f"✅ Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
else:
    print(f"❌ Model file not found: {MODEL_PATH}")

CLASS_NAMES = ["Healthy (Bình thường)", "Parkinson (Có dấu hiệu)"]

# ========================================
# IMAGE PREPROCESSING
# ========================================
def preprocess_spiral_image(image_pil):
    """
    Tiền xử lý ảnh xoắn ốc với các bước:
    1. Grayscale
    2. Noise Reduction (GaussianBlur + Bilateral Filter)
    3. CLAHE
    4. Adaptive Thresholding
    """
    # Convert to OpenCV format
    img_np = np.array(image_pil)
    if len(img_np.shape) == 3:
        if img_np.shape[2] == 4:  # RGBA
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
        else:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # 1. Grayscale
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY) if len(img_np.shape) == 3 else img_np
    
    # 2. Noise Reduction
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    bilateral = cv2.bilateralFilter(blur, 9, 75, 75)
    
    # 3. CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(bilateral)
    
    # 4. Adaptive Threshold
    thresh = cv2.adaptiveThreshold(
        clahe_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Convert back to RGB PIL
    processed = cv2.merge([thresh, thresh, thresh])
    processed_pil = Image.fromarray(processed)
    
    return processed_pil, thresh

# ========================================
# IMAGE PREDICTION
# ========================================
def predict_image_pil(image_pil):
    """Predict từ PIL Image"""
    if model is None:
        return None, 0.0, None
    
    try:
        # Preprocess
        processed_pil, _ = preprocess_spiral_image(image_pil)
        
        # Transform for model
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        img_tensor = transform(processed_pil).unsqueeze(0).to(DEVICE)
        
        # Predict
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
            
        return CLASS_NAMES[predicted.item()], confidence.item(), processed_pil
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, 0.0, None

# ========================================
# SYSTEM PROMPTS - OPTIMIZED
# ========================================
SYSTEM_PROMPT_CHAT = """Bạn là trợ lý AI y tế chuyên về bệnh Parkinson, được phát triển bởi nhóm học sinh và giáo viên trường THCS Nguyễn Tất Thành (Việt Nam).

## NHIỆM VỤ CHÍNH:
- Trả lời câu hỏi về bệnh Parkinson bằng tiếng Việt
- Cung cấp thông tin chính xác, dễ hiểu về Parkinson
- Hướng dẫn bài tập phục hồi chức năng cho bệnh nhân Parkinson
- Tư vấn chế độ sinh hoạt phù hợp

## XỬ LÝ CÂU HỎI KHÔNG LIÊN QUAN:
1. Nếu câu hỏi HOÀN TOÀN không liên quan đến sức khỏe/y tế:
   - Trả lời lịch sự: "Tôi là trợ lý chuyên về bệnh Parkinson. Câu hỏi này nằm ngoài phạm vi chuyên môn của tôi. Bạn có thắc mắc gì về Parkinson không?"

2. Nếu câu hỏi về BỆNH KHÁC (không phải Parkinson):
   - Trả lời ngắn gọn nếu biết, sau đó nhắc: "Tuy nhiên, chuyên môn của tôi là bệnh Parkinson. Nếu bạn có triệu chứng lạ hoặc lo lắng về sức khỏe, hãy theo dõi cơ thể và đến gặp bác sĩ để được tư vấn chính xác."

3. Nếu câu hỏi về bệnh có triệu chứng TƯƠNG TỰ Parkinson (run tay, run chân Essential Tremor, đa xơ cứng, đột quỵ...):
   - Giải thích sự khác biệt với Parkinson
   - Khuyên theo dõi sức khỏe và thăm khám bác sĩ Thần kinh để phân biệt chính xác

## QUY TẮC BẮT BUỘC:
1. KHÔNG BAO GIỜ hướng dẫn sử dụng thuốc hoặc đề cập tên thuốc cụ thể
2. LUÔN nhấn mạnh cần thăm khám bác sĩ chuyên khoa Thần kinh
3. Nếu người dùng hỏi về thuốc, trả lời: "Tôi không thể tư vấn về thuốc. Việc dùng thuốc cần bác sĩ chuyên khoa chỉ định và theo dõi."
4. Với triệu chứng nguy hiểm (ngã nhiều lần, khó nuốt, khó thở, sụt cân nhanh), khuyến cáo đến bệnh viện NGAY
5. Trả lời ngắn gọn, có cấu trúc rõ ràng
6. Nếu có triệu chứng lạ bất thường, luôn nhắc: "Hãy theo dõi sức khỏe và đến gặp bác sĩ nếu triệu chứng kéo dài hoặc nặng hơn."

## VIDEO BÀI TẬP PHỤC HỒI CÓ SẴN:
Khi người dùng hỏi về bài tập, hãy gợi ý CÁC VIDEO SAU (dùng đúng format để hiển thị iframe):

1. **Cải thiện cân bằng và dáng đi** - Giúp tăng cường thăng bằng, giảm nguy cơ té ngã
   [VIDEO:DS73cE9q79o:Bài tập cải thiện cân bằng và dáng đi]

2. **Đứng lên ngồi xuống tại giường** - Tăng cường sức mạnh cơ chân
   [VIDEO:bMcCqi6tllk:Bài tập đứng lên ngồi xuống]

3. **Xoay trở tại giường** - Cải thiện vận động khi nằm, giúp dễ trở mình
   [VIDEO:ND5eAvEREmg:Bài tập xoay trở tại giường]

## CÁCH GỢI Ý VIDEO:
- Khi người dùng hỏi về bài tập, CHỈ CẦN gợi ý 1-2 video phù hợp nhất
- Dùng ĐÚNG format: [VIDEO:VIDEO_ID:Tên bài tập]
- Giải thích ngắn gọn lợi ích của bài tập
- Nhắc nhở thực hiện an toàn, có người giám sát

## ĐỊNH DẠNG:
- Dùng bullet points khi liệt kê
- In đậm thông tin quan trọng
- Kết thúc bằng lời khuyên thăm khám nếu cần

## LƯU Ý:
Đây là công cụ sàng lọc hỗ trợ, KHÔNG thay thế chẩn đoán y khoa chuyên nghiệp."""

SYSTEM_PROMPT_COMBINED = """Bạn là chuyên gia AI phân tích dấu hiệu bệnh Parkinson, được phát triển bởi nhóm học sinh và giáo viên trường THCS Nguyễn Tất Thành.

## THÔNG TIN ĐẦU VÀO:
Bạn sẽ nhận được:
1. Kết quả phân tích ảnh xoắn ốc (nếu có)
2. Danh sách triệu chứng người dùng chọn

## NHIỆM VỤ:
Phân tích TỔNG HỢP các thông tin trên để đánh giá nguy cơ Parkinson.

## CẤU TRÚC TRẢ LỜI:

### 📊 ĐÁNH GIÁ TỔNG QUAN
[Mức độ nguy cơ: Thấp/Trung bình/Cao/Rất cao]
[Giải thích ngắn gọn lý do dựa trên CẢ ảnh và triệu chứng]

### 🔍 PHÂN TÍCH CHI TIẾT
**Về ảnh xoắn ốc:** [Nhận xét về độ đều, run tay, kiểm soát nét vẽ nếu có ảnh]
**Về triệu chứng:** [Phân tích từng nhóm triệu chứng đã chọn: vận động/phi vận động]

### ⚠️ CÁC DẤU HIỆU CẦN LƯU Ý
[Liệt kê triệu chứng đáng lo ngại, đặc biệt nếu có triệu chứng điển hình Parkinson]

### 💡 KHUYẾN NGHỊ
[Lời khuyên cụ thể dựa trên mức độ nguy cơ]
[Gợi ý bài tập phục hồi nếu phù hợp]

### 🏥 KHI NÀO CẦN GẶP BÁC SĨ
[Hướng dẫn cụ thể - LUÔN khuyến khích thăm khám dù nguy cơ thấp hay cao]

## QUY TẮC BẮT BUỘC:
1. KHÔNG đề cập thuốc hoặc tên thuốc cụ thể
2. LUÔN khuyến khích thăm khám bác sĩ chuyên khoa Thần kinh
3. Nhấn mạnh: "Đây là kết quả sàng lọc hỗ trợ, KHÔNG thay thế chẩn đoán của bác sĩ"
4. Với nguy cơ cao/rất cao, khuyến cáo khám BÁC SĨ NGAY
5. Nếu có triệu chứng nguy hiểm (ngã nhiều, khó nuốt, sụt cân nhanh): cảnh báo đậm và khuyên đi viện
6. Trả lời bằng tiếng Việt, rõ ràng, dễ hiểu, thân thiện
7. Luôn kết thúc bằng: "Hãy theo dõi sức khỏe và đến cơ sở y tế để được tư vấn chính xác nhất."""
# ========================================
# API ROUTES
# ========================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Chatbot endpoint"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({"error": "Vui lòng nhập câu hỏi."}), 400
        
        if not gemini_model:
            return jsonify({"error": "AI chưa được cấu hình."}), 500
        
        # Generate response
        prompt = f"{SYSTEM_PROMPT_CHAT}\n\n---\nCâu hỏi của người dùng: {user_message}\n---\nHãy trả lời:"
        
        response = gemini_model.generate_content(prompt)
        reply = response.text if response else "Xin lỗi, tôi không thể trả lời lúc này."
        
        # Format response
        reply = format_response_html(reply)
        
        return jsonify({"response": reply})
        
    except Exception as e:
        print(f"Chat error: {e}")
        return jsonify({"error": f"Lỗi xử lý: {str(e)}"}), 500

@app.route('/combined_analysis', methods=['POST'])
def combined_analysis():
    """Phân tích tổng hợp: Ảnh + Triệu chứng"""
    try:
        # Get symptoms
        import json
        symptoms_json = request.form.get('symptoms', '[]')
        symptoms = json.loads(symptoms_json)
        
        # Get image
        image_file = request.files.get('image')
        image_result = None
        confidence = 0
        image_base64 = None
        
        if image_file:
            image_pil = Image.open(image_file).convert('RGB')
            image_result, confidence, processed_pil = predict_image_pil(image_pil)
            
            # Convert to base64 for display
            buffered = io.BytesIO()
            processed_pil.save(buffered, format="JPEG")
            image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Build prompt
        analysis_input = ""
        
        if image_result:
            analysis_input += f"""
## KẾT QUẢ PHÂN TÍCH ẢNH XOẮN ỐC:
- Kết quả: {image_result}
- Độ tin cậy: {confidence * 100:.1f}%
"""
        else:
            analysis_input += "## KHÔNG CÓ ẢNH XOẮN ỐC\n"
        
        if symptoms:
            analysis_input += f"""
## TRIỆU CHỨNG NGƯỜI DÙNG CHỌN ({len(symptoms)} triệu chứng):
"""
            for i, symptom in enumerate(symptoms, 1):
                analysis_input += f"{i}. {symptom}\n"
        else:
            analysis_input += "## KHÔNG CÓ TRIỆU CHỨNG ĐƯỢC CHỌN\n"
        
        # Generate analysis
        if not gemini_model:
            return jsonify({"error": "AI chưa được cấu hình."}), 500
        
        full_prompt = f"{SYSTEM_PROMPT_COMBINED}\n\n{analysis_input}\n\nHãy phân tích tổng hợp:"
        
        response = gemini_model.generate_content(full_prompt)
        analysis = response.text if response else "Không thể phân tích lúc này."
        analysis = format_response_html(analysis)
        
        result = {
            "analysis": analysis,
            "symptoms_count": len(symptoms)
        }
        
        if image_result:
            result["image_result"] = image_result
            result["confidence"] = confidence
            result["image_base64"] = image_base64
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Combined analysis error: {e}")
        return jsonify({"error": f"Lỗi phân tích: {str(e)}"}), 500

# ========================================
# HELPER FUNCTIONS
# ========================================
def format_response_html(text):
    """Format markdown to HTML"""
    import re
    
    # Headers
    text = re.sub(r'^### (.+)$', r'<h4 class="font-bold text-teal-700 mt-3 mb-2">\1</h4>', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.+)$', r'<h3 class="font-bold text-lg text-teal-800 mt-4 mb-2">\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'^# (.+)$', r'<h2 class="font-bold text-xl text-teal-900 mt-4 mb-3">\1</h2>', text, flags=re.MULTILINE)
    
    # Bold
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    
    # Italic
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
    
    # Lists
    lines = text.split('\n')
    result = []
    in_list = False
    
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('- ') or stripped.startswith('• '):
            if not in_list:
                result.append('<ul class="list-disc pl-5 my-2 space-y-1">')
                in_list = True
            result.append(f'<li>{stripped[2:]}</li>')
        elif re.match(r'^\d+\. ', stripped):
            if not in_list:
                result.append('<ol class="list-decimal pl-5 my-2 space-y-1">')
                in_list = True
            li_content = re.sub(r'^\d+\. ', '', stripped)
            result.append(f'<li>{li_content}</li>')
        else:
            if in_list:
                result.append('</ul>' if result[-1] != '</ol>' else '</ol>')
                in_list = False
            if stripped:
                result.append(f'<p class="mb-2">{stripped}</p>')
            else:
                result.append('<br>')
    
    if in_list:
        result.append('</ul>')
    
    return '\n'.join(result)

# ========================================
# MAIN
# ========================================
if __name__ == '__main__':
    print("\n" + "="*60)
    print("🧠 HỆ THỐNG HỖ TRỢ PHÁT HIỆN SỚM PARKINSON")
    print("="*60)
    print(f"📊 Device: {DEVICE}")
    print(f"🤖 AI Model: {GEMINI_MODEL}")
    print(f"🔗 URL: http://127.0.0.1:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
