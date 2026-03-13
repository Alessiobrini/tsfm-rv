"""Move \\caption{} and \\label{} from above to below \\end{tabular} in all table files."""
import os
import re

TABLE_DIR = r'G:/Other computers/Dell Duke/Workfiles/Postdoc_file/human_x_AI_finance/paper/tables'

for fname in sorted(os.listdir(TABLE_DIR)):
    if not fname.endswith('.tex'):
        continue
    fpath = os.path.join(TABLE_DIR, fname)
    with open(fpath, 'r', encoding='utf-8') as f:
        content = f.read()

    tab_pos = content.find(r'\begin{tabular}')
    if tab_pos == -1:
        print(f'SKIP (no tabular): {fname}')
        continue

    before = content[:tab_pos]
    after = content[tab_pos:]

    # Find caption line in before section
    cap_pat = re.compile(r'\\caption\{.*?\}\n', re.DOTALL)
    lab_pat = re.compile(r'\\label\{[^}]*\}\n')

    cap_match = cap_pat.search(before)
    lab_match = lab_pat.search(before)

    if not cap_match:
        if '\\caption{' in after:
            print(f'SKIP (caption already below): {fname}')
        else:
            print(f'SKIP (no caption): {fname}')
        continue

    cap_text = cap_match.group(0).rstrip('\n')
    lab_text = lab_match.group(0).rstrip('\n') if lab_match else ''

    # Remove from before
    new_before = before.replace(cap_match.group(0), '')
    if lab_match:
        new_before = new_before.replace(lab_match.group(0), '')

    # Find \end{tabular}
    endtab_idx = after.find('\\end{tabular}')
    if endtab_idx == -1:
        print(f'SKIP (no end tabular): {fname}')
        continue

    endtab_end = endtab_idx + len('\\end{tabular}')
    before_end = after[:endtab_end]
    rest = after[endtab_end:]

    insert = '\n' + cap_text + '\n'
    if lab_text:
        insert += lab_text + '\n'

    new_content = new_before + before_end + insert + rest

    with open(fpath, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print(f'MOVED caption below: {fname}')
