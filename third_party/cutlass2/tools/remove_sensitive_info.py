import os
import re
import sys
import logging
import shutil
import pdb

def remove_examples(cutlass_root):
    examples = ["540_ppu_fp8_gemm",
                "541_ppu_fp4_gemm",
                "516_ppu_fp4_cute_gemm"]
    for example in examples:
        dirname = os.path.join(cutlass_root, "examples", example)
        if os.path.exists(dirname):
            if os.path.isdir(dirname):
                shutil.rmtree(dirname)
                print("Remove Dir: ", dirname)

    file_path = os.path.join(cutlass_root, "examples/CMakeLists.txt")
    if os.path.exists(file_path):
        output_lines = []
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            if not any(name in line for name in examples):
                output_lines.append(line)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)

def remove_file(file_path):
    tag_list = ['10500']
    for tag in tag_list:
        if tag in file_path:
            try:
                os.remove(file_path)
                print(f'Deleted: {file_path}')
                return True
            except Exception as e:
                print(f'Error deleting {file_path}: {e}')
                return False
    return False

'''
def remove_else_blocks_backup(file_path):
    patterns = [
        re.compile(r'#if ACOMPUTE_VERSION == 10000'),
        re.compile(r"#if\s+(.+?)\s+&&\s+ACOMPUTE_VERSION == 10000"),
        re.compile(r'#if !defined(ACOMPUTE_AIU_CONV)')
    ]
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    output_lines = []
    stack = []
    skip_mode = False

    for line in lines:
        stripped_line = line.strip()

        # Handle #if statements
        if stripped_line.startswith("#if"):
            # Check for specific version condition
            if any(pattern.match(stripped_line) for pattern in patterns):
                stack.append((skip_mode, True))  # Push current mode and mark that we have the target #if
                skip_mode = False  # Ensure not skipping inside a valid #if VERSION=100 block
            else:
                stack.append((skip_mode, False))  # Push the current mode and mark that it's not the target #if
            output_lines.append(line)

        elif stripped_line.startswith("#else") and stack:
            if stack[-1][1]:  # If last #if was the target condition
                skip_mode = True  # Start skipping lines
            output_lines.append(line)

        # Handle #endif statements
        elif stripped_line.startswith("#endif") and stack:
            skip_mode, is_target_if = stack.pop()
            if not skip_mode or not is_target_if:
                output_lines.append(line)

        # Copy non-skipped lines
        elif not skip_mode:
            output_lines.append(line)

    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(output_lines)
'''

def remove_if_blocks(lines):
    patterns = [
        re.compile(r'#if ACOMPUTE_VERSION == 10500'),
        re.compile(r"#if\s+(.+?)\s+&&\s+ACOMPUTE_VERSION == 10500"),
        re.compile(r'#if defined\(ACOMPUTE_AIU_CONV\)')
    ]

    output_lines = []
    skip_block = False
    nested_if_counter = 0

    for i in range(len(lines)):
        line = lines[i]
        stripped_line = line.strip()

        if (not skip_block) and any(pattern.match(stripped_line) for pattern in patterns):
            skip_block = True
            nested_if_counter = 0
            continue

        if skip_block:
            if stripped_line.startswith('#if'):
                nested_if_counter += 1
            elif stripped_line.startswith('#endif'):
                if nested_if_counter > 0:
                    nested_if_counter -= 1
                else:
                    skip_block = False
            elif stripped_line.startswith('#elif'):
                if nested_if_counter == 0:
                    output_lines.append(line.replace("#elif", "#if"))
                    skip_block = False
            elif stripped_line.startswith('#else'):
                if nested_if_counter == 0:
                    output_lines.append(line.replace("#else", "#if 1"))
                    skip_block = False
        else:
            output_lines.append(line)
    return output_lines

def remove_if_not_else_blocks(lines):
    output_lines = []
    skip_line = False
    depth = 0

    for line in lines:
        stripped_line = line.strip()
        if stripped_line.startswith("#if !defined(ACOMPUTE_AIU_CONV)"):
            depth += 1
            skip_line = True
        elif stripped_line.startswith("#else") and depth > 0:
            skip_line = False
        elif stripped_line.startswith("#endif") and depth > 0:
            depth -= 1
            if depth == 0:
                skip_line = False
        if not skip_line and depth == 0:
            output_lines.append(line)

    return output_lines

def remove_elif_blocks(lines):
    patterns = [
        re.compile(r'#elif ACOMPUTE_VERSION == 10500'),
        re.compile(r"#elif\s+(.+?)\s+&&\s+ACOMPUTE_VERSION == 10500")
    ]

    output_lines = []
    skip_block = False
    nested_if_counter = 0

    for i in range(len(lines)):
        line = lines[i]
        stripped_line = line.strip()

        if any(pattern.match(stripped_line) for pattern in patterns):
            skip_block = True
            nested_if_counter = 0
            continue

        if skip_block:
            if stripped_line.startswith('#if'):
                nested_if_counter += 1
            elif stripped_line.startswith('#endif'):
                if nested_if_counter > 0:
                    nested_if_counter -= 1
                else:
                    skip_block = False
                    output_lines.append(line)
            elif stripped_line.startswith('#elif'):
                if nested_if_counter == 0:
                    output_lines.append(line)
                    skip_block = False
            elif stripped_line.startswith('#else'):
                if nested_if_counter == 0:
                    output_lines.append(line)
                    skip_block = False
        else:
            output_lines.append(line)

    return output_lines

def remove_else_blocks(lines):
    patterns = [
        re.compile(r'#if ACOMPUTE_VERSION == 10000'),
        re.compile(r"#if\s+(.+?)\s+&&\s+ACOMPUTE_VERSION == 10000")
    ]

    result = []
    skip_line = False
    skip_stack = []
    skip_mode = False

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("#if"):
            if (not skip_mode) and any(pattern.match(line) for pattern in patterns):
                skip_mode = True
                result.append(lines[i])
            else:
                if not skip_line:
                    result.append(lines[i])
                skip_stack.append(skip_line)
        elif line.startswith("#else"):
            if skip_mode:
                skip_line = True
                skip_mode = False
            else:
                if not skip_line:
                    result.append(lines[i])
        elif line.startswith("#endif"):
            if skip_stack:
                skip_line = skip_stack.pop()
            else:
                skip_line = False
            if not skip_line:
                result.append(lines[i])
        else:
            if not skip_line:
                result.append(lines[i])
        i += 1
    return result

def remove_lines(lines):
    tags = ["10500"]

    output_lines = []
    for i in range(len(lines)):
        line = lines[i]
        if not any(tag in line for tag in tags):
            output_lines.append(line)

    return output_lines

def final_check(lines):
    tags = ["10500", "ACOMPUTE_AIU_CONV"]
    for i in range(len(lines)):
        line = lines[i]
        for tag in tags:
            if tag in line:
                print("Error, detect sensitive info: {}, line: {}, file: {}".format(tag, i, file_path))

def scan_repo_files(repo_path):
    print("====> start clean info !")
    remove_examples(repo_path)

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in ('build', 'docs', 'cmake', 'media', 'python')]
        for file in files:
            is_valid = file.endswith('.cpp') or file.endswith('.hpp') or file.endswith('.c') or file.endswith('.h') or file.endswith('.cu')
            if not is_valid: continue

            file_path = os.path.join(root, file)
            if os.path.islink(file_path): continue

            is_removed = remove_file(file_path)
            if is_removed: continue

            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            lines = remove_if_blocks(lines)
            lines = remove_if_not_else_blocks(lines)
            lines = remove_elif_blocks(lines)
            lines = remove_else_blocks(lines)
            lines = remove_lines(lines)

            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)

            final_check(lines)
    print("====> finish clean info !")


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        # cutlass/tools/__file
        path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if not os.path.exists(path):
        print(f"Cutlass not exists: {path}")
        return

    print(f"cutlass dir: {path}")

    scan_repo_files(path)

    # rm script
    os.remove(os.path.abspath(__file__))

if __name__ == "__main__":
    main()

