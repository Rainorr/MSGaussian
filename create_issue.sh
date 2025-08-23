#!/bin/bash

# 问题创建脚本
# 用法: ./create_issue.sh <问题编号> <问题简称>
# 示例: ./create_issue.sh 3 model-training

set -e

# 检查参数
if [ $# -ne 2 ]; then
    echo "用法: $0 <问题编号> <问题简称>"
    echo "示例: $0 3 model-training"
    exit 1
fi

ISSUE_NUM=$1
ISSUE_NAME=$2

# 格式化问题编号为三位数
ISSUE_NUM_FORMATTED=$(printf "%03d" $ISSUE_NUM)
ISSUE_DIR="docs/issues/issue-${ISSUE_NUM_FORMATTED}-${ISSUE_NAME}"

# 检查目录是否已存在
if [ -d "$ISSUE_DIR" ]; then
    echo "❌ 错误: 问题目录已存在: $ISSUE_DIR"
    exit 1
fi

# 创建目录结构
echo "📁 创建问题目录: $ISSUE_DIR"
mkdir -p "$ISSUE_DIR"/{files,logs,solution}

# 复制模板
echo "📝 创建问题文档..."
cp docs/templates/issue_template.md "$ISSUE_DIR/README.md"

# 替换模板中的占位符
sed -i "s/#XXX/#${ISSUE_NUM_FORMATTED}/g" "$ISSUE_DIR/README.md"
sed -i "s/\[YYYY-MM-DD\]/$(date +%Y-%m-%d)/g" "$ISSUE_DIR/README.md"

# 创建基本的日志文件
touch "$ISSUE_DIR/logs/error.log"
touch "$ISSUE_DIR/logs/debug.log"

# 更新问题索引
echo "📊 更新问题索引..."
INDEX_FILE="docs/issues/README.md"

# 在问题索引表格中添加新行
NEW_ROW="| [${ISSUE_NUM_FORMATTED}](issue-${ISSUE_NUM_FORMATTED}-${ISSUE_NAME}/) | ${ISSUE_NAME} | 🔄 进行中 | $(date +%Y-%m-%d) | - |"

# 使用sed在表格中插入新行
sed -i "/^| \[002\]/a\\$NEW_ROW" "$INDEX_FILE"

echo "✅ 问题创建完成!"
echo ""
echo "📋 下一步:"
echo "1. 编辑 $ISSUE_DIR/README.md 填写问题详情"
echo "2. 将相关文件复制到 $ISSUE_DIR/files/"
echo "3. 将错误日志保存到 $ISSUE_DIR/logs/"
echo "4. 完成后提交到Git"
echo ""
echo "🔧 快速编辑:"
echo "code $ISSUE_DIR/README.md"