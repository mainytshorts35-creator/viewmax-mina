import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="ViewMAX File Manager",
    page_icon="🖥️",
    layout="wide"
)

# Remove default Streamlit padding for full-screen view
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .block-container {
            padding: 0rem;
            max-width: 100%;
        }
    </style>
""", unsafe_allow_html=True)

# Render ViewMAX Interface inside Streamlit Component
viewmax_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: "Courier New", Courier, monospace, sans-serif; user-select: none; }
        body { background-color: #008080; height: 100vh; display: flex; flex-direction: column; overflow: hidden; }
        
        /* Top Menu Bar */
        .menu-bar { background: #c0c0c0; border-bottom: 2px solid #000; display: flex; padding: 2px 8px; font-size: 14px; font-weight: bold; }
        .menu-item { padding: 4px 12px; cursor: pointer; border: 1px solid transparent; }
        .menu-item:hover { background: #000080; color: #fff; }

        /* Main Workspace */
        .workspace { flex: 1; padding: 20px; display: flex; justify-content: center; align-items: center; }

        /* Window Frame */
        .window { background: #c0c0c0; border: 2px solid #fff; border-right-color: #404040; border-bottom-color: #404040; width: 100%; max-width: 780px; height: 480px; display: flex; flex-direction: column; box-shadow: 4px 4px 0px #000; }
        .title-bar { background: #000080; color: #fff; padding: 4px 8px; font-weight: bold; display: flex; justify-content: space-between; align-items: center; }
        .title-bar-buttons { display: flex; gap: 4px; }
        .btn-win { width: 16px; height: 14px; background: #c0c0c0; border: 1px solid #fff; border-right-color: #000; border-bottom-color: #000; font-size: 10px; line-height: 12px; text-align: center; cursor: pointer; color: #000; font-weight: bold; }

        /* Drive & Navigation Bar */
        .drive-bar { background: #c0c0c0; padding: 6px; border-bottom: 2px solid #808080; display: flex; gap: 8px; align-items: center; font-size: 13px; }
        .drive-btn { padding: 2px 10px; background: #c0c0c0; border: 2px solid #fff; border-right-color: #404040; border-bottom-color: #404040; cursor: pointer; font-weight: bold; }
        .drive-btn.active { border: 2px solid #404040; border-right-color: #fff; border-bottom-color: #fff; background: #a0a0a0; }
        .path-bar { background: #fff; border: 2px solid #808080; border-right-color: #fff; border-bottom-color: #fff; padding: 3px 6px; flex: 1; font-size: 13px; font-weight: bold; color: #000; }

        /* File Grid View */
        .window-body { flex: 1; background: #fff; border: 2px solid #808080; border-right-color: #fff; border-bottom-color: #fff; margin: 6px; padding: 12px; display: grid; grid-template-columns: repeat(auto-fill, minmax(110px, 1fr)); grid-auto-rows: 85px; gap: 12px; overflow-y: auto; }

        /* File Item Styling */
        .file-item { display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 6px; cursor: pointer; border: 1px dashed transparent; text-align: center; }
        .file-item:hover { border-color: #808080; background: #f0f0f0; }
        .file-item.selected { background: #000080; color: #fff; border-color: #000; }
        .icon { font-size: 26px; margin-bottom: 4px; }
        .label { font-size: 12px; word-break: break-all; }

        /* Status Bar */
        .status-bar { background: #c0c0c0; border-top: 2px solid #808080; padding: 4px 8px; font-size: 12px; display: flex; justify-content: space-between; }

        /* Modal File Viewer */
        .modal { display: none; position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 450px; height: 280px; background: #c0c0c0; border: 2px solid #fff; border-right-color: #000; border-bottom-color: #000; box-shadow: 6px 6px 0px #000; z-index: 100; flex-direction: column; }
        .modal-body { flex: 1; background: #fff; border: 2px solid #808080; margin: 8px; padding: 10px; font-size: 13px; white-space: pre-wrap; overflow-y: auto; color: #000; }
    </style>
</head>
<body>
    <div class="menu-bar">
        <div class="menu-item" onclick="alert('ViewMAX v1.0 - Streamlit Cloud')">Desk</div>
        <div class="menu-item" onclick="createNewFile()">New File</div>
        <div class="menu-item" onclick="goBack()">Up Directory</div>
        <div class="menu-item" onclick="location.reload()">Refresh</div>
    </div>

    <div class="workspace">
        <div class="window">
            <div class="title-bar">
                <span>ViewMAX Desktop Shell - [C:\]</span>
                <div class="title-bar-buttons">
                    <button class="btn-win" onclick="alert('Minimized')">_</button>
                    <button class="btn-win" onclick="alert('Maximized')">□</button>
                </div>
            </div>

            <div class="drive-bar">
                <span>Drives:</span>
                <button class="drive-btn" onclick="selectDrive('A')">A:</button>
                <button class="drive-btn active" id="btn-c" onclick="selectDrive('C')">C:</button>
                <div class="path-bar" id="current-path">C:\SYSTEM</div>
            </div>

            <div class="window-body" id="file-grid"></div>

            <div class="status-bar">
                <span id="status-text">6 Item(s)</span>
                <span>1,440 KB Available</span>
            </div>
        </div>
    </div>

    <div class="modal" id="viewer-modal">
        <div class="title-bar">
            <span id="modal-title">FILE.TXT</span>
            <button class="btn-win" onclick="closeModal()">X</button>
        </div>
        <div class="modal-body" id="modal-content"></div>
    </div>

    <script>
        const fileSystem = {
            'C': {
                'SYSTEM': [
                    { name: '..', type: 'folder' },
                    { name: 'DOS', type: 'folder' },
                    { name: 'GAMES', type: 'folder' },
                    { name: 'VIEWMAX.EXE', type: 'app', content: 'Binary executable file.' },
                    { name: 'AUTOEXEC.BAT', type: 'file', content: '@ECHO OFF\\nPROMPT $P$G\\nPATH C:\\\\DOS;C:\\\\SYSTEM\\nSET TEMP=C:\\\\TEMP' },
                    { name: 'CONFIG.SYS', type: 'file', content: 'FILES=40\\nBUFFERS=20\\nDEVICE=C:\\\\DOS\\\\HIMEM.SYS' },
                    { name: 'README.TXT', type: 'file', content: 'Welcome to ViewMAX Streamlit Edition!' }
                ],
                'DOS': [
                    { name: '..', type: 'folder' },
                    { name: 'COMMAND.COM', type: 'app', content: 'DOS Command Processor.' }
                ],
                'GAMES': [
                    { name: '..', type: 'folder' },
                    { name: 'DOOM.EXE', type: 'app', content: 'Starting DOOM...' }
                ]
            },
            'A': {
                'ROOT': [
                    { name: 'FLOPPY.TXT', type: 'file', content: 'Data stored on Floppy Disk A:' }
                ]
            }
        };

        let currentDrive = 'C';
        let currentFolder = 'SYSTEM';
        let selectedElement = null;

        function renderFiles() {
            const grid = document.getElementById('file-grid');
            const pathDisplay = document.getElementById('current-path');
            const statusDisplay = document.getElementById('status-text');
            
            grid.innerHTML = '';
            pathDisplay.innerText = `${currentDrive}:\\${currentFolder === 'ROOT' ? '' : currentFolder}`;

            const items = fileSystem[currentDrive][currentFolder] || [];
            statusDisplay.innerText = `${items.length} Item(s)`;

            items.forEach(item => {
                const el = document.createElement('div');
                el.className = 'file-item';

                let icon = '📄';
                if (item.type === 'folder') icon = '📁';
                if (item.type === 'app') icon = '⚙️';

                el.innerHTML = `<div class="icon">${icon}</div><div class="label">${item.name}</div>`;

                el.onclick = (e) => {
                    e.stopPropagation();
                    if (selectedElement) selectedElement.classList.remove('selected');
                    el.classList.add('selected');
                    selectedElement = el;
                };

                el.ondblclick = () => {
                    if (item.type === 'folder') {
                        if (item.name === '..') {
                            goBack();
                        } else {
                            currentFolder = item.name;
                            renderFiles();
                        }
                    } else {
                        openFile(item.name, item.content);
                    }
                };

                grid.appendChild(el);
            });
        }

        function selectDrive(drive) {
            currentDrive = drive;
            currentFolder = drive === 'C' ? 'SYSTEM' : 'ROOT';
            document.querySelectorAll('.drive-btn').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
            renderFiles();
        }

        function goBack() {
            if (currentFolder !== 'SYSTEM' && currentFolder !== 'ROOT') {
                currentFolder = 'SYSTEM';
                renderFiles();
            }
        }

        function openFile(name, content) {
            document.getElementById('modal-title').innerText = name;
            document.getElementById('modal-content').innerText = content || 'File is empty.';
            document.getElementById('viewer-modal').style.display = 'flex';
        }

        function closeModal() {
            document.getElementById('viewer-modal').style.display = 'none';
        }

        function createNewFile() {
            const fileName = prompt('Enter File Name (e.g., TEST.TXT):', 'NEWFILE.TXT');
            if (fileName) {
                fileSystem[currentDrive][currentFolder].push({
                    name: fileName.toUpperCase(),
                    type: 'file',
                    content: 'User created document.'
                });
                renderFiles();
            }
        }

        renderFiles();
    </script>
</body>
</html>
"""

components.html(viewmax_html, height=620)
