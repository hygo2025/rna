# Versão do Python a ser usada pelo pyenv
PYTHON_VERSION = 3.13
# Diretório do ambiente virtual
VENV_DIR = .venv
# Arquivo de dependências
REQUIREMENTS_FILE = requirements.txt

# Caminhos para os executáveis dentro do venv no Windows
VENV_PYTHON = $(VENV_DIR)\Scripts\python.exe
VENV_PIP = $(VENV_DIR)\Scripts\pip.exe

.PHONY: install update clean remove check_python_version

# Regra principal para instalar tudo
install: $(VENV_DIR)

# Verifica se o pyenv-win tem a versão do Python instalada.
# Simplificamos a lógica para ser mais robusta no Windows.
# O comando `pyenv install` é seguro de ser executado mesmo que a versão já exista.
check_python_version:
	@echo "Verificando se o Python $(PYTHON_VERSION) esta instalado via pyenv..."
	@pyenv versions | findstr /C:"$(PYTHON_VERSION)" > nul 2>&1 || ( \
		echo "Python $(PYTHON_VERSION) nao encontrado. Instalando..." && \
		pyenv install $(PYTHON_VERSION) \
	)
	@echo "Python $(PYTHON_VERSION) esta disponivel."
	@pyenv global $(PYTHON_VERSION)
	@pyenv local $(PYTHON_VERSION)

# Cria o ambiente virtual e instala as dependências
$(VENV_DIR): check_python_version
	@echo "Criando ambiente virtual em $(VENV_DIR)..."
	@python -m venv $(VENV_DIR)
	@echo "Instalando/Atualizando pip e dependencias de $(REQUIREMENTS_FILE)..."
	@$(VENV_PYTHON) -m pip install --upgrade pip
	@$(VENV_PIP) install -r $(REQUIREMENTS_FILE)

# Atualiza as dependências do projeto
update:
	@$(VENV_PIP) install --upgrade -r $(REQUIREMENTS_FILE)
	@echo "Dependencias atualizadas."

# Limpa o ambiente virtual e outros artefatos
clean:
	@if exist $(VENV_DIR) ( \
		echo "Removendo diretorio $(VENV_DIR)..." && \
		rmdir /s /q $(VENV_DIR) \
	)
	@if exist recommendations ( \
		echo "Removendo diretorio recommendations..." && \
		rmdir /s /q recommendations \
	)

# Limpa tudo e remove o arquivo de dependências
remove: clean
	@if exist $(REQUIREMENTS_FILE) ( \
		del /f $(REQUIREMENTS_FILE) \
	)