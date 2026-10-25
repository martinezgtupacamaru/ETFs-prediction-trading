"""Execute actual notebook cells in a fresh Python process; save outputs after each cell."""
import os
import sys
import time
from pathlib import Path
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
import matplotlib
from matplotlib_inline.backend_inline import configure_inline_support

if __name__ == '__main__':
    path = Path(__file__).resolve().parent/'research.ipynb'
    os.chdir(path.parent)
    sys.path.insert(0,str(path.parent))
    notebook = nbformat.read(path,as_version=4)
    matplotlib.use('module://matplotlib_inline.backend_inline')
    shell = InteractiveShell.instance()
    configure_inline_support(shell,'inline')
    count = 0
    for i,cell in enumerate(notebook.cells):
        if cell.cell_type != 'code':
            continue
        count += 1
        started = time.time()
        print('START cell',i,flush=True)
        with capture_output() as captured:
            result = shell.run_cell(cell.source,store_history=True)
        cell.execution_count = count
        cell.outputs = []
        for name,value in [('stdout',captured.stdout),('stderr',captured.stderr)]:
            if value:
                cell.outputs.append(nbformat.v4.new_output('stream',name=name,text=value))
        for out in captured.outputs:
            cell.outputs.append(nbformat.v4.new_output('display_data',data=out.data,metadata=out.metadata))
        nbformat.write(notebook,path)
        print(captured.stdout,flush=True)
        print('END cell',i,'seconds',round(time.time()-started,2),flush=True)
        if result.error_before_exec or result.error_in_exec:
            raise RuntimeError(f'Cell {i}: {result.error_before_exec or result.error_in_exec}')
    nbformat.validate(notebook)
    print('Notebook executed successfully.',flush=True)
