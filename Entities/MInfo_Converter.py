import os
import subprocess
import re
import json
import sys
import shutil
import copy
import time
import struct

from .ModelInfo import ModelInfo
from .ModelSkeleton import ModelSkeleton
# GBFR Blender .json export to .minfo converter
# Version 3.0
# By AlphaSatanOmega - https://github.com/AlphaSatanOmega
# Drag and drop the original .minfo and the Blender export .json onto this .py file

MINFO_HEADER = bytes.fromhex("50000000000000004800640060005C005800540050004C004800440034000000")
SKELETON_HEADER = bytes.fromhex("100000000C0010000C000800060004000C000000")
BUFFER_TYPE_WEIGHT_INDICES = 2
BUFFER_TYPE_SECONDARY_WEIGHTS = 4
BUFFER_TYPE_WEIGHTS = 8

# Convert flatc json data string to proper json data string with quotes
def preprocess_flatbuffers_json(json_data):
    return re.sub(r'(\w+)(?=\s*:)', r'"\1"', json_data) # Use regular expression to wrap field names in quotes

def is_valid_minfo(filepath):
    try:
        with open(filepath, 'rb') as file:
            return file.read(32) == MINFO_HEADER
    except Exception:
        return False

def is_valid_skeleton(filepath):
    try:
        with open(filepath, 'rb') as file:
            return file.read(20) == SKELETON_HEADER
    except Exception:
        return False

def decode_name(value):
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)

def vec3_to_dict(vec):
    if vec is None:
        return None
    return {"x": vec.X(), "y": vec.Y(), "z": vec.Z()}

def quat_to_dict(quat):
    if quat is None:
        return None
    return {"w": quat.W(), "x": quat.X(), "y": quat.Y(), "z": quat.Z()}

def bbox_to_dict(bbox):
    if bbox is None:
        return None
    from .Vec3 import Vec3
    min_vec = bbox.Min(Vec3())
    max_vec = bbox.Max(Vec3())
    return {"min": vec3_to_dict(min_vec), "max": vec3_to_dict(max_vec)}

def dump_json(out_json_path, data):
    with open(out_json_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=2)

def map_deform_joint_indices(indices, deform_joint_table):
    bone_ids = []
    out_of_range_indices = []

    for index in indices:
        if index < len(deform_joint_table):
            bone_ids.append(deform_joint_table[index])
        else:
            bone_ids.append(None)
            out_of_range_indices.append(index)

    return bone_ids, out_of_range_indices

def write_minfo_json(out_json_path, source_json_path):
    with open(source_json_path, 'r', encoding='utf-8') as file:
        dump_json(out_json_path, json.load(file))

def write_skeleton_json(skeleton_path, out_json_path):
    if not os.path.exists(skeleton_path) or not is_valid_skeleton(skeleton_path):
        return

    with open(skeleton_path, 'rb') as file:
        skeleton = ModelSkeleton.GetRootAs(bytearray(file.read()), 0)

    bones = []
    for index in range(skeleton.BodyLength()):
        bone = skeleton.Body(index)
        bone_info = bone.A1()
        bones.append({
            "index": index,
            "name": decode_name(bone.Name()),
            "parent_id": bone.ParentId(),
            "bone_info": None if bone_info is None else {
                "bone_id": bone_info.BoneId(),
                "unk": bone_info.Unk()
            },
            "position": vec3_to_dict(bone.Position()),
            "rotation": quat_to_dict(bone.Quat()),
            "scale": vec3_to_dict(bone.Scale())
        })

    dump_json(out_json_path, {
        "magic": skeleton.Magic(),
        "bone_count": skeleton.BodyLength(),
        "bones": bones
    })

def write_mmesh_json(minfo_path, mmesh_path, out_json_path):
    if not os.path.exists(minfo_path) or not os.path.exists(mmesh_path) or not is_valid_minfo(minfo_path):
        return

    with open(minfo_path, 'rb') as file:
        model_info = ModelInfo.GetRootAs(bytearray(file.read()), 0)

    lod = model_info.Lodinfos(0)
    if lod is None:
        return

    vert_count = lod.VertCount()
    face_count = lod.PolyCountX3() // 3
    deform_joint_table = [model_info.BonesToWeightIndices(i) for i in range(model_info.BonesToWeightIndicesLength())]

    mesh_buffers = []
    for i in range(lod.MeshBuffersLength()):
        mesh_buffer = lod.MeshBuffers(i)
        mesh_buffers.append({
            "index": i,
            "offset": mesh_buffer.Offset(),
            "size": mesh_buffer.Size()
        })

    chunks = []
    for i in range(lod.ChunksLength()):
        chunk = lod.Chunks(i)
        chunks.append({
            "index": i,
            "offset": chunk.Offset(),
            "count": chunk.Count(),
            "sub_mesh_id": chunk.SubMesh(),
            "material_id": chunk.Material(),
            "a5": chunk.Unk1(),
            "a6": chunk.Unk2()
        })

    sub_meshes = []
    for i in range(model_info.SubMeshesLength()):
        sub_mesh = model_info.SubMeshes(i)
        sub_meshes.append({
            "index": i,
            "name": decode_name(sub_mesh.Name()),
            "bbox": bbox_to_dict(sub_mesh.Bbox())
        })

    vertices = []
    weight_indices = []
    weights = []
    faces = []

    with open(mmesh_path, 'rb') as file:
        if lod.MeshBuffersLength() > 0:
            file.seek(lod.MeshBuffers(0).Offset())

        for index in range(vert_count):
            vertex_buffer = file.read(32)
            position = struct.unpack('<fff', vertex_buffer[0:12])
            normal = struct.unpack('<eee', vertex_buffer[12:18])
            tangent_data = struct.unpack('<eeee', vertex_buffer[20:28])
            tangent = tangent_data[:3]
            bitangent_sign = tangent_data[3]
            uv = struct.unpack('<ee', vertex_buffer[28:32])

            vertices.append({
                "index": index,
                "position": [position[0], position[1], position[2]],
                "normal": [normal[0], normal[1], normal[2]],
                "tangent": [tangent[0], tangent[1], tangent[2]],
                "bitangent_sign": bitangent_sign,
                "uv": [uv[0], uv[1]]
            })

        if lod.BufferTypes() & BUFFER_TYPE_WEIGHT_INDICES:
            file.seek(lod.MeshBuffers(1).Offset())
            for _ in range(vert_count):
                indices = list(struct.unpack('<HHHH', file.read(8)))
                bone_ids, out_of_range_indices = map_deform_joint_indices(indices, deform_joint_table)
                weight_indices.append({
                    "raw": indices,
                    "bone_ids": bone_ids,
                    "out_of_range_indices": out_of_range_indices
                })

        if lod.BufferTypes() & BUFFER_TYPE_WEIGHTS:
            weight_buffer_index = 3 if lod.BufferTypes() & BUFFER_TYPE_SECONDARY_WEIGHTS else 2
            file.seek(lod.MeshBuffers(weight_buffer_index).Offset())
            for _ in range(vert_count):
                raw_weights = struct.unpack('<HHHH', file.read(8))
                weights.append({
                    "raw": list(raw_weights),
                    "normalized": [value / 65535 for value in raw_weights]
                })

        if lod.MeshBuffersLength() > 0:
            file.seek(lod.MeshBuffers(lod.MeshBuffersLength() - 1).Offset())
            for face_index in range(face_count):
                face = struct.unpack('<III', file.read(12))
                faces.append({
                    "index": face_index,
                    "vertices": [face[0], face[1], face[2]]
                })

    dump_json(out_json_path, {
        "vertex_count": vert_count,
        "face_count": face_count,
        "buffer_types": lod.BufferTypes(),
        "mesh_buffers": mesh_buffers,
        "chunks": chunks,
        "sub_meshes": sub_meshes,
        "bones_to_weight_indices": deform_joint_table,
        "vertices": vertices,
        "weight_indices": weight_indices,
        "weights": weights,
        "faces": faces
    })

def write_export_json_files(output_dir, model_name):
    minfo_json_path = os.path.join(output_dir, f"{model_name}.json")
    write_minfo_json(os.path.join(output_dir, f"{model_name}.minfo.json"), minfo_json_path)
    write_skeleton_json(
        os.path.join(output_dir, f"{model_name}.skeleton"),
        os.path.join(output_dir, f"{model_name}.skeleton.json")
    )
    write_mmesh_json(
        os.path.join(output_dir, f"{model_name}.minfo"),
        os.path.join(output_dir, f"{model_name}.mmesh"),
        os.path.join(output_dir, f"{model_name}.mmesh.json")
    )

def replace_mesh_info(flatc_json, blender_json, magic = None):
    # Load json data from files
    flatc_json_data = json.loads(flatc_json)
    blender_json_data = json.loads(blender_json)

    if magic: # Overwrite magic if number provided
        flatc_json_data["magic"] = magic

    # Replace the mesh info in the flatc json with the mesh info from the blender export json
    keys_to_replace = ["mesh_buffers", "chunks", "vertex_count", "poly_count_x3", "buffer_types"]
    for lod_index in range(len(flatc_json_data["lods"])):
        for key in keys_to_replace:
            flatc_json_data["lods"][lod_index][key] = blender_json_data[key]
    # Just set the lods array to contain the Highest LOD
    flatc_json_data["lods"] = [flatc_json_data["lods"][0]]

    # Replace the bones_to_weight_indices list with the blender export list
    flatc_json_data["bones_to_weight_indices"] = blender_json_data["bones_to_weight_indices"]

    # Replace Sub meshes
    flatc_json_data["sub_meshes"] = blender_json_data["sub_meshes"]

    """
    # Get submesh names
    blender_json_submesh_names = blender_json_data["SubMeshes"]
    flatc_json_submesh_names = []
    for flatc_submesh in flatc_json_data["SubMeshes"]:
        flatc_json_submesh_names.append(flatc_submesh["Name"])
    # If a submesh name from blender is not in the flatc submeshes, 
    # duplicate the last submesh and change its name to match
    for blender_submesh_name in blender_json_submesh_names:
        if blender_submesh_name not in flatc_json_submesh_names:
            new_submesh = copy.deepcopy(flatc_json_data["SubMeshes"][-1])
            new_submesh["Name"] = blender_submesh_name
            flatc_json_data["SubMeshes"].append(new_submesh)
    """

    return json.dumps(flatc_json_data, indent=2) # Convert and return

def convert_minfo(flatc_path, minfo_path, blender_json_path, magic = None):
    print ("Start MInfo Conversion.")
    
    if os.path.dirname(minfo_path) != os.path.dirname(blender_json_path):
        raise Exception("\n\nERROR: A copy of the .minfo needs to be in same location as you are exporting to.")
    
    script_dir = os.path.dirname(__file__) # Get the script directory
    export_dir = os.path.dirname(blender_json_path) # Get blender export directory
    flatc_temp_dir = os.path.join(export_dir, "_flatc_temp")
    minfo_fbs_path = os.path.join(script_dir,"MInfo_ModelInfo.fbs") # Get the FlatBuffers Schema
#    flatc_path = os.path.join(script_dir, "flatc.exe") # Get the path to flatc.exe in the same directory
    model_name = os.path.splitext(os.path.basename(minfo_path))[0] # Get the model name from the minfo
    
    # Generate json from .minfo file using flatc.exe

    print(flatc_path, "-o", f"{flatc_temp_dir}", "--json", f"{minfo_fbs_path}", "--", f"{minfo_path}", "--raw-binary")
    
    command = [flatc_path, "-o", f"{flatc_temp_dir}", "--json", f"{minfo_fbs_path}", "--", f"{minfo_path}", "--raw-binary", "--no-warnings"]
    subprocess.run(command, check=True)
    # flatc json gets stored to a temp folder
    flatc_json_path = os.path.join(flatc_temp_dir, f"{model_name}.json") 
    print(f"Generated: {flatc_json_path}")
    
    # Open the json files
    with open(flatc_json_path, 'r') as flatc_file, open(blender_json_path, 'r') as blender_file:
        flatc_json = flatc_file.read()
        blender_json = blender_file.read()
        flatc_json = preprocess_flatbuffers_json(flatc_json) # Fix flatc json
        
    # Replace the mesh info of flatc json with blender export json's mesh info
    modified_flatc_json = replace_mesh_info(flatc_json, blender_json, magic)
    # print(modified_flatc_json)
    # Save modified flatc to a file in the same directory as the script
    # os.path.join(export_dir, f"{model_name}.json")
    modified_flatc_json_file = blender_json_path # Overwrite blender json file
    with open(modified_flatc_json_file, 'w') as file:
        file.write(modified_flatc_json)
    print(f"Replaced mesh info in {flatc_json_path} with mesh info from {minfo_path}")

    # Create output directory next to original .minfo
    output_dir = os.path.join(os.path.dirname(minfo_path), "_Exported_MInfo")
    os.makedirs(output_dir, exist_ok=True)

    # Run flatc.exe to generate binary minfo from the modified json
    command = [flatc_path, "-o", f"{flatc_temp_dir}", "--binary", f"{minfo_fbs_path}", modified_flatc_json_file, "--no-warnings"]
    subprocess.run(command, check=True)
    # Rename the .bin otuput to .minfo
    binary_output_file = os.path.join(flatc_temp_dir, f"{model_name}.bin")
    minfo_output_file = binary_output_file.replace(".bin", '.minfo')
    os.rename(binary_output_file, minfo_output_file)
    print(f"Modified {minfo_output_file} generated.")
    # Move minfo to output_dir
    try: os.remove(os.path.join(output_dir, f"{model_name}.minfo")) # Remove copy if exists
    except: print(f"No copy of {model_name}.minfo found in {output_dir}, moving.")
    shutil.move(minfo_output_file, output_dir)
    
    # Move all the Blender export files into the output_dir
    blender_export_file_exts = [".mmesh", ".skeleton", ".json"]
    for file_ext in blender_export_file_exts:
        original_file_path = os.path.join(export_dir, f"{model_name}{file_ext}")
        try: os.remove(os.path.join(output_dir, f"{model_name}{file_ext}")) # Remove copy if exists
        except Exception as e: 
            print(str(e))
            print(f"No copy of {model_name}{file_ext} found in {output_dir}, moving.")
        shutil.move(original_file_path, output_dir)

    write_export_json_files(output_dir, model_name)
         
    # Remove _flatc_temp safely
    os.remove(os.path.join(flatc_temp_dir, f"{model_name}.json")) # Remove json first
    os.rmdir(flatc_temp_dir) # Then delete the folder since it should be empty, fails otherwise
    print(f"Removed {flatc_temp_dir}")
    
    print(f"Modified JSON and binary files moved to: {output_dir}")


def main():
    input_file_1 = sys.argv[1]
    input_file_2 = sys.argv[2]
    # Check which file is the .minfo and which is the .json
    if input_file_1.lower().endswith('.minfo') and input_file_2.lower().endswith('.json'):
        minfo_path = input_file_1
        blender_json_path = input_file_2
    elif input_file_1.lower().endswith('.json') and input_file_2.lower().endswith('.minfo'):
        minfo_path = input_file_2
        blender_json_path = input_file_1
    else:
        print("Error: Incorrect files input. The inputs should be an .minfo and a .json file.")
    # print(minfo_path, blender_json_path)

    # Process the files
    try:
        convert_minfo(minfo_path, blender_json_path)
        print("\nConversion Complete!")
    except Exception as e:
        print(str(e))

if __name__ == "__main__":
    main()
    input("\nPress any key to exit...")
