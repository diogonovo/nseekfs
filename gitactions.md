# 🔐 GitHub Secrets Setup Guide

Este guia explica como configurar os secrets necessários para o pipeline CI/CD funcionar corretamente.

## 📋 Secrets Necessários

### 🧪 Test PyPI (Ambiente: `test-pypi`)

1. **TEST_PYPI_API_TOKEN**
   - **Onde obter**: https://test.pypi.org/manage/account/token/
   - **Permissões**: "Scope to entire account" ou específico para o projeto
   - **Formato**: `pypi-...` (o token completo)

### 🚀 Production PyPI (Ambiente: `pypi`)

2. **PYPI_API_TOKEN**
   - **Onde obter**: https://pypi.org/manage/account/token/
   - **Permissões**: "Scope to entire account" ou específico para o projeto nseekfs
   - **Formato**: `pypi-...` (o token completo)

### 🔧 GitHub (Automático)

3. **GITHUB_TOKEN**
   - ✅ **Automático**: GitHub fornece automaticamente
   - ❌ **Não configurar manualmente**

## 🛠️ Como Configurar os Secrets

### Passo 1: Acessar Configurações do Repositório

1. Vá para: `https://github.com/diogonovo/nseekfs/settings`
2. No menu lateral: **Secrets and variables** → **Actions**

### Passo 2: Criar Environments

#### Environment: `test-pypi`
1. Clique em **Environments** (se não existir)
2. **New environment** → Nome: `test-pypi`
3. **Add secret**:
   - Name: `TEST_PYPI_API_TOKEN`
   - Value: `pypi-...` (seu token do test.pypi.org)

#### Environment: `pypi`
1. **New environment** → Nome: `pypi`
2. **Add secret**:
   - Name: `PYPI_API_TOKEN`
   - Value: `pypi-...` (seu token do pypi.org)
3. ⚠️ **Configurar proteção**:
   - ✅ **Required reviewers**: `diogonovo`
   - ✅ **Wait timer**: 0 minutes
   - ✅ **Deployment branches**: Only protected branches

### Passo 3: Verificar Configuração

Execute este comando para verificar se está tudo configurado:

```bash
# Verificar se environments existem
curl -H "Authorization: token $GITHUB_TOKEN" \
     "https://api.github.com/repos/diogonovo/nseekfs/environments"
```

## 🔑 Como Obter os Tokens PyPI

### Test PyPI Token

1. Acesse: https://test.pypi.org/manage/account/token/
2. **Create token**
3. **Token name**: `nseekfs-github-actions`
4. **Scope**: 
   - ✅ Entire account (recomendado para começar)
   - ⚪ Specific project: `nseekfs` (se já existir)
5. **Create token**
6. ⚠️ **Copie o token**: `pypi-AgEI...` (só aparece uma vez!)

### Production PyPI Token

1. Acesse: https://pypi.org/manage/account/token/
2. **Create token**
3. **Token name**: `nseekfs-github-actions-prod`
4. **Scope**:
   - ⚪ Entire account
   - ✅ **Specific project**: `nseekfs` (depois do primeiro upload)
5. **Create token**
6. ⚠️ **Copie o token**: `pypi-AgEI...`

## 🔒 Configuração de Segurança

### Proteção do Environment `pypi`

Para proteger releases de produção:

```yaml
# .github/environments/pypi.yml (configurar via UI)
protection_rules:
  - type: required_reviewers
    reviewers:
      - diogonovo
  - type: wait_timer
    wait_timer: 0
  - type: branch_policy
    branch_policy: 
      protected_branches: true
      custom_branches: []
```

### Permissões Mínimas

Os tokens devem ter apenas as permissões mínimas necessárias:

- ✅ **upload packages**
- ❌ ~~delete packages~~
- ❌ ~~manage account~~

## 🧪 Testar Configuração

### 1. Trigger Manual Release

```bash
# Criar uma release de teste
git tag v0.1.0-test
git push origin v0.1.0-test
```

### 2. Verificar Logs

Monitore os workflows em:
- https://github.com/diogonovo/nseekfs/actions

### 3. Verificar Test PyPI

Após upload bem-sucedido:
- https://test.pypi.org/project/nseekfs/

## ❗ Troubleshooting

### Erro: "Invalid or non-existent authentication information"

```bash
# Verificar se token está correto
# Token deve começar com 'pypi-'
# Não incluir espaços ou caracteres extras
```

### Erro: "Repository not found"

```bash
# Para PyPI production, o projeto deve existir primeiro
# Faça primeiro upload manual ou via Test PyPI
```

### Erro: "Insufficient permissions"

```bash
# Token precisa ter permissão de upload
# Para projetos existentes, usar scope "specific project"
```

## 🔄 Rotação de Tokens

### Cronograma Recomendado

- 🔄 **Rotacionar a cada 6 meses**
- 🚨 **Rotacionar imediatamente se comprometido**

### Processo de Rotação

1. Gerar novo token no PyPI
2. Atualizar secret no GitHub
3. Testar com release de desenvolvimento
4. Revogar token antigo

## 📞 Suporte

Se tiver problemas:

1. **Verificar logs** do GitHub Actions
2. **Testar tokens** manualmente com `twine`
3. **Consultar documentação**:
   - [PyPI API tokens](https://pypi.org/help/#apitoken)
   - [GitHub Secrets](https://docs.github.com/en/actions/security-guides/encrypted-secrets)

---

**⚠️ Importante**: Nunca compartilhe ou commite tokens no código! Sempre use GitHub Secrets.