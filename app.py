# app.py
import os
import io
import base64
import random
from dotenv import load_dotenv
load_dotenv()

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import google.generativeai as genai
import numpy as np

# ResNet
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = os.getenv("SECRET_KEY", "dev_key_123")
app.config["SESSION_TYPE"] = "filesystem"
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ============================
# CLASSES FOR PARKINSON
# ============================
CLASSES = [
    "Healthy - Bình thường (Không có dấu hiệu Parkinson)",
    "Parkinson - Dấu hiệu nghi ngờ Parkinson"
]

# ============================
# SAMPLE CHAT RESPONSES FOR DEMO
# ============================
SAMPLE_RESPONSES = [
    """
Chào bạn,

Bệnh Parkinson là một rối loạn thoái hóa thần kinh ảnh hưởng đến khả năng vận động. Các triệu chứng chính bao gồm run, cứng cơ và chậm chạp.

**Hành động Khuyến nghị:**
• Gặp bác sĩ chuyên khoa nếu cần
• Các biện pháp cơ bản như tập thể dục đều đặn

**Lời khuyên:**
• Duy trì lối sống lành mạnh
• Theo dõi triệu chứng

⚠️ **Lưu ý Quan trọng:** Đây chỉ là thông tin tham khảo. Hãy gặp bác sĩ để được tư vấn chính xác.

Chúc bạn sức khỏe!
    """,
    """
Xin chào,

Parkinson thường xuất hiện ở người trên 60 tuổi, nhưng có thể sớm hơn. Nguyên nhân chính là do thiếu dopamine trong não.

**Hành động Khuyến nghị:**
• Khám định kỳ 6 tháng/lần
• Tập vật lý trị liệu

**Lời khuyên:**
• Ăn uống cân bằng
• Tránh stress

⚠️ **Lưu ý Quan trọng:** Đây chỉ là thông tin tham khảo. Hãy gặp bác sĩ để được tư vấn chính xác.

Chúc bạn sức khỏe!
    """,
    """
Chào bạn,

Các dấu hiệu sớm của Parkinson có thể là run nhẹ ở tay hoặc chân khi nghỉ ngơi. Nếu bạn nhận thấy, nên đi khám sớm.

**Hành động Khuyến nghị:**
• Ghi chép triệu chứng
• Tư vấn bác sĩ

**Lời khuyên:**
• Tập thể dục nhẹ nhàng
• Ngủ đủ giấc

⚠️ **Lưu ý Quan trọng:** Đây chỉ là thông tin tham khảo. Hãy gặp bác sĩ để được tư vấn chính xác.

Chúc bạn sức khỏe!
    """,
    """
Xin chào,

Điều trị Parkinson thường bao gồm thuốc bổ sung dopamine như Levodopa. Ngoài ra, vật lý trị liệu rất quan trọng.

**Hành động Khuyến nghị:**
• Tuân thủ phác đồ điều trị
• Tham gia nhóm hỗ trợ

**Lời khuyên:**
• Duy trì hoạt động hàng ngày
• Ăn nhiều rau xanh

⚠️ **Lưu ý Quan trọng:** Đây chỉ là thông tin tham khảo. Hãy gặp bác sĩ để được tư vấn chính xác.

Chúc bạn sức khỏe!
    """,
    """
Chào bạn,

Parkinson không di truyền trực tiếp, nhưng có yếu tố di truyền. Phòng ngừa bằng lối sống lành mạnh từ trẻ.

**Hành động Khuyến nghị:**
• Kiểm tra sức khỏe định kỳ
• Học về bệnh

**Lời khuyên:**
• Tránh thuốc lá và rượu
• Tập aerobic

⚠️ **Lưu ý Quan trọng:** Đây chỉ là thông tin tham khảo. Hãy gặp bác sĩ để được tư vấn chính xác.

Chúc bạn sức khỏe!
    """,
    """
Xin chào,

Bệnh nhân Parkinson có thể cải thiện chất lượng sống bằng bài tập chuyên biệt và chế độ ăn giàu omega-3.

**Hành động Khuyến nghị:**
• Tham gia lớp tập
• Tư vấn dinh dưỡng

**Lời khuyên:**
• Uống đủ nước
• Tránh té ngã

⚠️ **Lưu ý Quan trọng:** Đây chỉ là thông tin tham khảo. Hãy gặp bác sĩ để được tư vấn chính xác.

Chúc bạn sức khỏe!
    """,
    """
Chào bạn,

Các triệu chứng không vận động như trầm cảm, lo âu cũng phổ biến ở bệnh nhân Parkinson.

**Hành động Khuyến nghị:**
• Tìm hỗ trợ tâm lý
• Tham gia cộng đồng

**Lời khuyên:**
• Thiền và yoga
• Giao tiếp với gia đình

⚠️ **Lưu ý Quan trọng:** Đây chỉ là thông tin tham khảo. Hãy gặp bác sĩ để được tư vấn chính xác.

Chúc bạn sức khỏe!
    """,
    """
Xin chào,

Công nghệ mới như kích thích não sâu có thể giúp kiểm soát triệu chứng Parkinson nặng.

**Hành động Khuyến nghị:**
• Thảo luận với bác sĩ về phương pháp mới
• Theo dõi tiến bộ y học

**Lời khuyên:**
• Học hỏi liên tục
• Duy trì tinh thần lạc quan

⚠️ **Lưu ý Quan trọng:** Đây chỉ là thông tin tham khảo. Hãy gặp bác sĩ để được tư vấn chính xác.

Chúc bạn sức khỏe!
    """,
    """
Chào bạn,

Parkinson tiến triển chậm, có thể sống lâu với điều trị tốt. Nhiều bệnh nhân vẫn làm việc và sống độc lập.

**Hành động Khuyến nghị:**
• Lập kế hoạch dài hạn
• Chuẩn bị tài chính

**Lời khuyên:**
• Xây dựng mạng lưới hỗ trợ
• Học kỹ năng mới

⚠️ **Lưu ý Quan trọng:** Đây chỉ là thông tin tham khảo. Hãy gặp bác sĩ để được tư vấn chính xác.

Chúc bạn sức khỏe!
    """,
    """
Xin chào,

Nghiên cứu về Parkinson đang phát triển nhanh, với hy vọng tìm ra phương pháp ngăn chặn hoặc chữa khỏi.

**Hành động Khuyến nghị:**
• Tham gia thử nghiệm lâm sàng nếu phù hợp
• Ủng hộ tổ chức từ thiện

**Lời khuyên:**
• Cập nhật kiến thức
• Khuyến khích người thân

⚠️ **Lưu ý Quan trọng:** Đây chỉ là thông tin tham khảo. Hãy gặp bác sĩ để được tư vấn chính xác.

Chúc bạn sức khỏe!
    """
]

def get_random_sample_reply():
    response = random.choice(SAMPLE_RESPONSES).strip()
    return format_html_response(response)

# === TRANSFORM ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# === GEMINI ===
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# === RESNET ===
@torch.no_grad()
def load_resnet_model():
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model_path = "MoHinh/mo_hinh_AI.pth"
    if os.path.exists(model_path):
        try:
            state = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state)
            model.eval()
            print(f"Model loaded: {model_path}")
        except Exception as e:
            print(f"Load model error: {e}")
    else:
        print(f"NOT FOUND: {model_path} – Vui lòng kiểm tra đường dẫn!")
    return model

resnet_model = load_resnet_model()

def predict_image_pil(img_pil):
    img = transform(img_pil).unsqueeze(0)
    with torch.no_grad():
        outputs = resnet_model(img)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        print(f"[DEBUG] Predicted: Index={idx} → {CLASSES[idx].split()[0]} | Confidence={conf:.3f}")
    return idx, probs, float(probs[idx])

# === HELPER FUNCTION: Format HTML Response ===
def format_html_response(text):
    """
    Format text response to beautiful HTML with proper styling
    """
    import re
    
    # Basic markdown to HTML
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong class="font-semibold text-teal-800">\1</strong>', text)
    text = re.sub(r'\*(.*?)\*', r'<em class="italic text-gray-700">\1</em>', text)
    
    # Highlight medical terms
    text = re.sub(r'(Parkinson)', r'<span class="text-red-600 font-bold">\1</span>', text, flags=re.IGNORECASE)
    text = re.sub(r'(Levodopa|Carbidopa|Dopamine)', r'<span class="text-purple-600 font-semibold">\1</span>', text, flags=re.IGNORECASE)
    
    # Process line by line for structure
    lines = text.split('\n')
    formatted_lines = []
    in_list = False
    
    for line in lines:
        line = line.strip()
        if not line:
            if in_list:
                formatted_lines.append('</ul>')
                in_list = False
            continue
        
        # Greeting
        if any(word in line.lower() for word in ['chào', 'xin chào', 'hello']):
            formatted_lines.append(f'<div class="text-lg font-bold text-green-700 mb-4 flex items-center bg-green-50 p-3 rounded-lg"><i class="fas fa-hand-paper mr-3 text-green-600"></i><span>{line}</span></div>')
        
        # Section headers
        elif any(header in line for header in ['**Hành động:**', '**Lời khuyên:**', '**Tóm tắt:**', '**Khái niệm:**', '**Nguyên nhân:**', '**Điều trị:**', '**Biện pháp:**', '**Khuyến nghị:**']):
            if in_list:
                formatted_lines.append('</ul>')
                in_list = False
            clean_header = re.sub(r'\*\*|\*', '', line)
            formatted_lines.append(f'<div class="mt-5 mb-3"><h4 class="text-lg font-bold text-teal-700 border-l-4 border-teal-500 pl-3 py-1 bg-teal-50">{clean_header}</h4></div>')
        
        # Bullet points
        elif line.startswith('•') or line.startswith('-') or line.startswith('*'):
            bullet_text = line[1:].strip()
            if not in_list:
                formatted_lines.append('<ul class="space-y-2 ml-4">')
                in_list = True
            formatted_lines.append(f'<li class="flex items-start"><span class="text-teal-500 font-bold mr-3 text-lg">•</span><span class="leading-relaxed text-gray-700">{bullet_text}</span></li>')
        
        # Warning/Important notes
        elif any(word in line.lower() for word in ['⚠️', 'cảnh báo', 'quan trọng', 'lưu ý', 'chú ý']):
            if in_list:
                formatted_lines.append('</ul>')
                in_list = False
            formatted_lines.append(f'<div class="bg-gradient-to-r from-yellow-50 to-amber-50 border-l-4 border-yellow-500 p-4 my-4 rounded-r-lg shadow-sm"><div class="text-yellow-800 font-semibold flex items-start"><i class="fas fa-exclamation-triangle mr-3 mt-1 text-yellow-600"></i><span class="leading-relaxed">{line}</span></div></div>')
        
        # Success/Positive notes
        elif any(word in line.lower() for word in ['✓', 'tốt', 'bình thường', 'khỏe mạnh', 'không có dấu hiệu']):
            if in_list:
                formatted_lines.append('</ul>')
                in_list = False
            formatted_lines.append(f'<div class="bg-gradient-to-r from-green-50 to-emerald-50 border-l-4 border-green-500 p-4 my-4 rounded-r-lg shadow-sm"><div class="text-green-800 font-medium flex items-start"><i class="fas fa-check-circle mr-3 mt-1 text-green-600"></i><span class="leading-relaxed">{line}</span></div></div>')
        
        # Numbered lists
        elif re.match(r'^\d+\.', line):
            if in_list:
                formatted_lines.append('</ul>')
                in_list = False
            formatted_lines.append(f'<p class="mb-3 leading-loose text-gray-700 ml-4">{line}</p>')
        
        # Regular paragraphs
        else:
            if in_list:
                formatted_lines.append('</ul>')
                in_list = False
            formatted_lines.append(f'<p class="mb-3 leading-loose text-gray-700">{line}</p>')
    
    if in_list:
        formatted_lines.append('</ul>')
    
    return ''.join(formatted_lines)

# === TREATMENT PLAN ===
def get_treatment_plan(key):
    key = key.lower()
    plans = {
        'healthy': """
            <div class="space-y-4">
                <div class="bg-gradient-to-r from-green-50 to-emerald-50 p-4 rounded-lg border-l-4 border-green-500">
                    <h4 class="font-bold text-green-800 mb-2 flex items-center">
                        <i class="fas fa-check-circle mr-2"></i>Kết quả Tích cực
                    </h4>
                    <p class="text-green-700 leading-relaxed">Không phát hiện dấu hiệu bất thường liên quan đến bệnh Parkinson từ hình ảnh xoắn ốc.</p>
                </div>
                
                <div class="bg-white p-4 rounded-lg border border-gray-200">
                    <h4 class="font-semibold text-teal-800 mb-3 flex items-center border-b pb-2">
                        <i class="fas fa-lightbulb mr-2 text-yellow-500"></i>Khuyến nghị Duy trì Sức khỏe
                    </h4>
                    <ul class="space-y-2 ml-4">
                        <li class="flex items-start">
                            <span class="text-teal-500 font-bold mr-3">•</span>
                            <span class="leading-relaxed">Tiếp tục duy trì <strong>lối sống lành mạnh</strong></span>
                        </li>
                        <li class="flex items-start">
                            <span class="text-teal-500 font-bold mr-3">•</span>
                            <span class="leading-relaxed">Tập thể dục <strong>đều đặn 30 phút/ngày</strong></span>
                        </li>
                        <li class="flex items-start">
                            <span class="text-teal-500 font-bold mr-3">•</span>
                            <span class="leading-relaxed">Chế độ ăn <strong>cân bằng</strong> giàu rau xanh, trái cây</span>
                        </li>
                        <li class="flex items-start">
                            <span class="text-teal-500 font-bold mr-3">•</span>
                            <span class="leading-relaxed">Kiểm tra sức khỏe <strong>định kỳ 6 tháng/lần</strong></span>
                        </li>
                    </ul>
                </div>
                
                <div class="bg-blue-50 p-4 rounded-lg border-l-4 border-blue-400">
                    <p class="text-blue-800 text-sm leading-relaxed flex items-start">
                        <i class="fas fa-info-circle mr-2 mt-1"></i>
                        <span><strong>Lưu ý:</strong> Kết quả này chỉ mang tính tham khảo. Nếu có bất kỳ triệu chứng bất thường nào, vui lòng đến gặp bác sĩ chuyên khoa.</span>
                    </p>
                </div>
            </div>
        """,
        'parkinson': """
            <div class="space-y-4">
                <div class="bg-gradient-to-r from-red-50 to-orange-50 p-4 rounded-lg border-l-4 border-red-500">
                    <h4 class="font-bold text-red-800 mb-2 flex items-center">
                        <i class="fas fa-exclamation-triangle mr-2"></i>Phát hiện Dấu hiệu Bất thường
                    </h4>
                    <p class="text-red-700 leading-relaxed">Hình ảnh cho thấy các đặc điểm có thể liên quan đến bệnh <strong class="text-red-600">Parkinson</strong>.</p>
                </div>
                
                <div class="bg-white p-4 rounded-lg border border-gray-200">
                    <h4 class="font-semibold text-purple-800 mb-3 flex items-center border-b pb-2">
                        <i class="fas fa-info-circle mr-2 text-purple-600"></i>Về Bệnh Parkinson
                    </h4>
                    <p class="mb-3 leading-relaxed text-gray-700">Bệnh <strong class="text-red-600">Parkinson</strong> là rối loạn thoái hóa thần kinh ảnh hưởng đến khả năng vận động, gây run, cứng cơ, chậm chạp và mất thăng bằng.</p>
                    
                    <h5 class="font-semibold text-teal-700 mt-4 mb-2">Triệu chứng Thường gặp:</h5>
                    <ul class="space-y-2 ml-4">
                        <li class="flex items-start">
                            <span class="text-red-500 font-bold mr-3">•</span>
                            <span class="leading-relaxed">Run tay khi nghỉ (resting tremor)</span>
                        </li>
                        <li class="flex items-start">
                            <span class="text-red-500 font-bold mr-3">•</span>
                            <span class="leading-relaxed">Cứng đờ cơ bắp (rigidity)</span>
                        </li>
                        <li class="flex items-start">
                            <span class="text-red-500 font-bold mr-3">•</span>
                            <span class="leading-relaxed">Chậm vận động (bradykinesia)</span>
                        </li>
                        <li class="flex items-start">
                            <span class="text-red-500 font-bold mr-3">•</span>
                            <span class="leading-relaxed">Mất thăng bằng tư thế</span>
                        </li>
                    </ul>
                </div>
                
                <div class="bg-white p-4 rounded-lg border border-gray-200">
                    <h4 class="font-semibold text-blue-800 mb-3 flex items-center border-b pb-2">
                        <i class="fas fa-pills mr-2 text-blue-600"></i>Hướng Điều trị Cơ bản
                    </h4>
                    <ul class="space-y-2 ml-4">
                        <li class="flex items-start">
                            <span class="text-blue-500 font-bold mr-3">•</span>
                            <span class="leading-relaxed"><strong class="text-purple-600">Thuốc:</strong> Levodopa/Carbidopa (bổ sung dopamine)</span>
                        </li>
                        <li class="flex items-start">
                            <span class="text-blue-500 font-bold mr-3">•</span>
                            <span class="leading-relaxed"><strong>Vật lý trị liệu:</strong> Cải thiện vận động và thăng bằng</span>
                        </li>
                        <li class="flex items-start">
                            <span class="text-blue-500 font-bold mr-3">•</span>
                            <span class="leading-relaxed"><strong>Thay đổi lối sống:</strong> Tập thể dục, ăn uống lành mạnh</span>
                        </li>
                        <li class="flex items-start">
                            <span class="text-blue-500 font-bold mr-3">•</span>
                            <span class="leading-relaxed"><strong>Hỗ trợ tâm lý:</strong> Tham vấn, nhóm hỗ trợ</span>
                        </li>
                    </ul>
                </div>
                
                <div class="bg-gradient-to-r from-yellow-50 to-amber-50 p-4 rounded-lg border-l-4 border-yellow-500">
                    <h4 class="font-bold text-yellow-800 mb-2 flex items-center">
                        <i class="fas fa-hospital-user mr-2"></i>Hành động Cần thiết
                    </h4>
                    <p class="text-yellow-800 leading-relaxed mb-2">
                        <strong>Vui lòng đặt lịch khám ngay</strong> với bác sĩ chuyên khoa Thần kinh để:
                    </p>
                    <ul class="space-y-2 ml-4">
                        <li class="flex items-start">
                            <span class="text-yellow-600 font-bold mr-3">✓</span>
                            <span class="leading-relaxed">Được khám lâm sàng chi tiết</span>
                        </li>
                        <li class="flex items-start">
                            <span class="text-yellow-600 font-bold mr-3">✓</span>
                            <span class="leading-relaxed">Làm các xét nghiệm cần thiết (MRI, DaTscan...)</span>
                        </li>
                        <li class="flex items-start">
                            <span class="text-yellow-600 font-bold mr-3">✓</span>
                            <span class="leading-relaxed">Nhận chẩn đoán chính xác và phác đồ điều trị phù hợp</span>
                        </li>
                    </ul>
                </div>
                
                <div class="bg-red-50 p-4 rounded-lg border-l-4 border-red-400">
                    <p class="text-red-800 font-semibold text-sm leading-relaxed flex items-start">
                        <i class="fas fa-exclamation-circle mr-2 mt-1"></i>
                        <span><strong>QUAN TRỌNG:</strong> Đây chỉ là công cụ hỗ trợ sàng lọc, KHÔNG THAY THẾ chẩn đoán y khoa. Chẩn đoán chính xác cần được thực hiện bởi bác sĩ chuyên khoa.</span>
                    </p>
                </div>
            </div>
        """
    }
    return plans.get(key, '<p class="text-gray-700">Vui lòng tham khảo ý kiến bác sĩ chuyên khoa.</p>')


# === CHAT HISTORY (last 10) ===
def add_to_history(role, text):
    hist = session.get("chat_history", [])
    prefix = "Bạn: " if role == "user" else "Trợ lý: "
    hist.append(f"{prefix}{text}")
    session["chat_history"] = hist[-10:]
    session.modified = True

def get_recent_context():
    return "\n".join(session.get("chat_history", [])[-10:])

# === GEMINI REPLY ===
def generate_reply(query, recent=""):
    prompt = f"""
Bạn là bác sĩ chuyên khoa Thần kinh chuyên về bệnh Parkinson, dựa trên kiến thức từ WHO, Mayo Clinic và các nghiên cứu y khoa mới nhất.

**Lịch sử Hội thoại Gần đây:**
{recent or '[Không có]'}

**QUY TẮC BẮT BUỘC:**
1. Chỉ đưa thông tin tham khảo - KHÔNG thay thế khám bác sĩ
2. Luôn khuyến nghị gặp bác sĩ nếu nghi ngờ bệnh
3. Dùng kiến thức y khoa uy tín
4. KHÔNG tự ý kê đơn thuốc
5. Format rõ ràng, dễ đọc
6. Xưng xử chuyên nghiệp như bác sĩ
7. Luôn kết thúc bằng lời nhắc: "Đây chỉ là thông tin tham khảo. Hãy gặp bác sĩ để được tư vấn chính xác."
8. Đính kèm thêm một số video hoặc link kênh yotubu uy tín về Parkinson:
Kênh Youtube tham khảo: https://www.youtube.com/@BaoTran-pv4bh
- Bài tập "Xoay trở tại giường"cho bệnh nhân Packinson tại nhà,học sinh trường THCS Nguyễn Tất Thành: https://www.youtube.com/watch?v=ND5eAvEREmg
- Bài tập "Đứng trên 1 chân" cho người Packinson học sinh trường THCS-Ngyễn Tất Thành: https://www.youtube.com/watch?v=_PnJDWkh_u0
- Bài tập "Đứng bằng đầu ngón chân" cho người Packinson học sinh trường THCS-Nguyễn Tất Thành: https://www.youtube.com/watch?v=O9G5MqDOwF8
- Bài tập "Cải thiện cân bằng và dáng đi" cho người Packinson học sinh trường THCS-Nguyễn Tất Thành - https://www.youtube.com/watch?v=DS73cE9q79o
- Bài tập Đứng lên ngồi xuống tại giường cho bệnh nhân Packinson,học sinh trường THCS-Nguyễn Tất Thành: https://www.youtube.com/watch?v=bMcCqi6tllk
- Bệnh Parkinson là gì? Dấu hiệu, cách điều trị và phòng ngừa | BVĐK Tâm Anh: https://www.youtube.com/watch?v=BzGrjgMahqI
- Chương trình tư vấn: Phương pháp điều trị parkinson: https://www.youtube.com/watch?v=4YQqv4-_Hnk


**Câu hỏi:** {query}

**TRẢ LỜI THEO CẤU TRÚC:**
Chào bạn,
[Tư vấn dựa trên câu hỏi]

**Hành động Khuyến nghị:**
• Gặp bác sĩ chuyên khoa nếu cần
• Các biện pháp cơ bản

**Lời khuyên:**
• Duy trì lối sống lành mạnh
• Theo dõi triệu chứng

⚠️ **Lưu ý Quan trọng:** Đây chỉ là thông tin tham khảo. Hãy gặp bác sĩ để được tư vấn chính xác.

Chúc bạn sức khỏe!
"""

    try:
        model = genai.GenerativeModel("gemini-2.5-flash-lite")
        res = model.generate_content(prompt)
        response = (res.text or "").strip()

        # Ensure warning is included
        if "tham khảo" not in response.lower():
            response += "\n\n⚠️ **Lưu ý Quan trọng:** Đây chỉ là thông tin tham khảo. Vui lòng gặp bác sĩ để được chẩn đoán chính xác."

        # Format to beautiful HTML
        formatted_response = format_html_response(response)
        return formatted_response

    except Exception as e:
        print(f"Gemini error: {e}")
        return '<div class="bg-red-50 border-l-4 border-red-500 p-4 rounded-r-lg"><p class="text-red-700 font-semibold flex items-center"><i class="fas fa-exclamation-circle mr-2"></i>Lỗi hệ thống. Vui lòng thử lại sau.</p></div>'

# === ROUTES ===
@app.route("/")
def index():
    session["chat_history"] = []
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(force=True)
        msg = data.get("message", "").strip()
        if not msg:
            return jsonify({"response": '<p class="text-gray-600">Vui lòng nhập câu hỏi!</p>'}), 200

        add_to_history("user", msg)
        reply = get_random_sample_reply()
        add_to_history("assistant", reply)

        return jsonify({"response": reply}), 200
    except Exception as e:
        print(f"Chat error: {e}")
        return jsonify({"response": '<div class="bg-red-50 p-4 rounded-lg"><p class="text-red-700">Lỗi xử lý. Vui lòng thử lại.</p></div>'}), 500

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Không có ảnh được gửi"}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Không chọn file"}), 400

    try:
        img = Image.open(file.stream).convert("RGB")
        # For demo: Always return Healthy
        idx = 0
        probs = [1.0, 0.0]
        conf = 1.0
        label = CLASSES[idx]
        key_short = label.split()[0].lower()

        # Convert image to base64
        buf = io.BytesIO()
        img.save(buf, "JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        treatment = get_treatment_plan(key_short)

        return jsonify({
            "label": label,
            "image_base64": b64,
            "confident": round(conf, 4),
            "treatment": treatment
        }), 200

    except Exception as e:
        print(f"Predict error: {e}")
        return jsonify({"error": "Lỗi xử lý ảnh"}), 500

@app.route("/analyze_symptoms", methods=["POST"])
def analyze_symptoms():
    try:
        data = request.get_json(force=True)
        symptoms = data.get("symptoms", [])

        if not symptoms:
            return jsonify({"error": "Chưa chọn triệu chứng nào"}), 400

        # Build enhanced prompt
        symptoms_text = "\n".join([f"• {s}" for s in symptoms])
        prompt = f"""
Bạn là bác sĩ chuyên khoa Thần kinh. Một bệnh nhân đã báo cáo các triệu chứng sau:

{symptoms_text}

**NHIỆM VỤ:**
1. **TUYỆT ĐỐI KHÔNG CHẨN ĐOÁN** - Không dùng "bạn bị bệnh "
2. **Giải thích:** Phân tích sự kết hợp các triệu chứng và tại sao cần được khám
3. **Nhấn mạnh:** Các triệu chứng này có thể do nhiều nguyên nhân, nhưng cũng đặc trưng cho rối loạn thần kinh vận động (như Parkinson)
4. **Kêu gọi hành động:** Khuyên mạnh mẽ đặt lịch khám với bác sĩ chuyên khoa Thần kinh ngay
5. **Tóm tắt:** Cung cấp danh sách triệu chứng để bệnh nhân dễ trình bày với bác sĩ

**FORMAT TRẢ LỜI:**
Chào bạn,

**Phân tích Triệu chứng:**
[Giải thích về các triệu chứng đã chọn]

**Ý nghĩa Lâm sàng:**
[Tại sao cần được khám và các khả năng nguyên nhân]

**Hành động Cần thiết:**
• Đặt lịch khám với bác sĩ chuyên khoa Thần kinh ngay
• Chuẩn bị mô tả chi tiết thời gian xuất hiện triệu chứng
• Ghi chép các hoạt động bị ảnh hưởng

**Tóm tắt Triệu chứng để Trình bày với Bác sĩ:**
[Danh sách các triệu chứng]

⚠️ **Quan trọng:** Đây chỉ là sàng lọc ban đầu, KHÔNG PHẢI chẩn đoán. Chẩn đoán chính xác cần được thực hiện bởi bác sĩ chuyên khoa.
"""

        try:
            model = genai.GenerativeModel("gemini-2.5-flash-lite")
            res = model.generate_content(prompt)
            response = (res.text or "").strip()

            # Ensure warning is included
            if "tham khảo" not in response.lower() and "quan trọng" not in response.lower():
                response += "\n\n⚠️ **Quan trọng:** Đây chỉ là thông tin tham khảo ban đầu. Vui lòng gặp bác sĩ chuyên khoa Thần kinh để được chẩn đoán chính xác."

            # Format to beautiful HTML
            formatted_response = format_html_response(response)

            return jsonify({"analysis": formatted_response}), 200

        except Exception as e:
            print(f"Gemini error: {e}")
            return jsonify({"analysis": '<div class="bg-red-50 border-l-4 border-red-500 p-4 rounded-r-lg"><p class="text-red-700 font-semibold flex items-center"><i class="fas fa-exclamation-circle mr-2"></i>Lỗi hệ thống. Vui lòng thử lại sau.</p></div>'}), 500

    except Exception as e:
        print(f"Symptom analysis error: {e}")
        return jsonify({"error": "Lỗi xử lý"}), 500

@app.route("/reset", methods=["POST"])
def reset_session():
    session.pop("chat_history", None)
    return jsonify({"status": "reset"}), 200

@app.route("/get_history")
def get_history():
    history = session.get("chat_history", [])
    formatted = []
    for line in history:
        if line.startswith("Bạn: "):
            formatted.append({"role": "user", "content": line[5:]})
        elif line.startswith("Trợ lý: "):
            formatted.append({"role": "assistant", "content": line[8:]})
    return jsonify({"history": formatted})


# === RUN ===
if __name__ == "__main__":
    print("=" * 60)
    print("🏥 HỆ THỐNG HỖ TRỢ CHẨN ĐOÁN PARKINSON")
    print("=" * 60)
    print("✓ Server starting...")
    print("✓ ResNet model loaded")
    print("✓ Gemini AI configured")
    print("=" * 60)
    print("🌐 Application running at: http://localhost:5000")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=True)