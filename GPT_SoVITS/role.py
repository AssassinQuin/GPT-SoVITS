import os
import json


def gen_role(role_json_path):
    # 读取role.json文件，处理可能存在的UTF-8 BOM
    with open(role_json_path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    role_map = {}

    def get_files(path):
        files = os.listdir(path)

        def find_file(extension):
            try:
                return next(f for f in files if f.endswith(extension))
            except StopIteration:
                raise FileNotFoundError(f"No {extension} file found in {path}")

        wav_file = find_file(".wav")
        pth_file = find_file(".pth")
        ckpt_file = find_file(".ckpt")

        return {
            "ref_audio_path": os.path.join(path, wav_file),
            "ref_text": os.path.splitext(wav_file)[0],  # wav 文件名（不包括扩展名）
            "gpt_model": os.path.join(path, ckpt_file),
            "sovits_model": os.path.join(path, pth_file),
        }

    # 遍历所有角色并保存数据到role_map
    for role_category, role_list in data.items():
        if isinstance(role_list, list):
            for role in role_list:
                role_name = role["name"]
                role_path = role["path"]
                try:
                    role_map[role_name] = get_files(role_path)
                    role_map[role_name]["role"] = role_category
                except FileNotFoundError as e:
                    print(f"Error processing {role_name}: {e}")
        else:  # 针对女主和男主这种单独的角色
            role_name = role_list["name"]
            role_path = role_list["path"]
            try:
                role_map[role_name] = get_files(role_path)
                role_map[role_name]["role"] = role_category
            except FileNotFoundError as e:
                print(f"Error processing {role_name}: {e}")

    # # 打印或保存role_map以供验证
    # print(json.dumps(role_map, ensure_ascii=False, indent=4))

    # # 或者保存到一个新的JSON文件中
    # with open("role_map.json", "w", encoding="utf-8") as f:
    #     json.dump(role_map, f, ensure_ascii=False, indent=4)

    return role_map


if __name__ == "__main__":
    role_json_path = "/root/code/GPT-SoVITS/model/role.json"
    role_map = gen_role(role_json_path)
    print(json.dumps(role_map, ensure_ascii=False, indent=4))
