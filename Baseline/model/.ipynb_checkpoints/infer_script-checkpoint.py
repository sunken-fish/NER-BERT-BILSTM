# 快速查看输出格式是否满足要求，看生成的res.txt是否符合要求
import os
from customize_service import CustomizeService


# 待预测文件路径
test_path = "data/test.txt"
service = CustomizeService(model_name="model.pth", model_path=".")

post_data = {"input_txt": {os.path.basename(test_path): open(test_path, "rb")}}
print('post_data:',post_data)
data = service._preprocess(post_data)
data = service._inference(data)
data = service._postprocess(data)

with open("res.txt", "w", encoding="utf-8") as f:
    f.writelines(data.get("result"))
