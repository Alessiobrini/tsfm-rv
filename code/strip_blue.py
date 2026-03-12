"""Remove all \\textcolor{blue}{...} wrappers from main.tex, keeping content."""
import re

path = r'G:/Other computers/Dell Duke/Workfiles/Postdoc_file/human_x_AI_finance/paper/main.tex'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

def remove_textcolor_blue(text):
    result = []
    i = 0
    marker = '\\textcolor{blue}{'
    while i < len(text):
        if text[i:i+len(marker)] == marker:
            i += len(marker)
            depth = 1
            start = i
            while i < len(text) and depth > 0:
                ch = text[i]
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                if depth > 0:
                    i += 1
            # text[start:i] is the content
            result.append(text[start:i])
            i += 1  # skip closing }
        else:
            result.append(text[i])
            i += 1
    return ''.join(result)

new_text = remove_textcolor_blue(text)

count_before = text.count('\\textcolor{blue}')
count_after = new_text.count('\\textcolor{blue}')
print(f'Before: {count_before}, After: {count_after}')

with open(path, 'w', encoding='utf-8') as f:
    f.write(new_text)
print('Done')
