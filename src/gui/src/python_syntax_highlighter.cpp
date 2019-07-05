//
// Created by egi on 7/5/19.
//

#include "python_syntax_highlighter.h"


python_syntax_highlighter::python_syntax_highlighter (QTextDocument *parent)
 : QSyntaxHighlighter(parent)
{
  const QString keyword_patterns[] = {
      QStringLiteral("\\bfor\\b"), QStringLiteral("\\bif\\b"), QStringLiteral("\\band\\b"),
      QStringLiteral("\\bbreak\\b"), QStringLiteral("\\bclass\\b"), QStringLiteral("\\bcontinue\\b"),
      QStringLiteral("\\bdef\\b"), QStringLiteral("\\bdel\\b"), QStringLiteral("\\belif\\b"),
      QStringLiteral("\\belse\\b"), QStringLiteral("\\bexcept\\b"), QStringLiteral("\\bexec\\b"),
      QStringLiteral("\\bis\\b"), QStringLiteral("\\blambda\\b"), QStringLiteral("\\bnot\\b"),
      QStringLiteral("\\bor\\b"), QStringLiteral("\\bpass\\b"), QStringLiteral("\\braise\\b"),
      QStringLiteral("\\breturn\\b"), QStringLiteral("\\btry\\b"), QStringLiteral("\\bwhile\\b"),
      QStringLiteral("\\byield\\b"), QStringLiteral("\\bNone\\b"), QStringLiteral("\\bTrue\\b"),
      QStringLiteral("\\bFalse\\b"), QStringLiteral("\\bfrom\\b"), QStringLiteral("\\bimport\\b"),
      QStringLiteral("\\bprint\\b")
  };

  const QString operators_patterns[] = {
      QStringLiteral("=="), QStringLiteral(">"), QStringLiteral("<"), QStringLiteral("="),
      QStringLiteral(">="), QStringLiteral("<="), QStringLiteral("!="), QStringLiteral("\\+"),
      QStringLiteral("-"), QStringLiteral("\\+="), QStringLiteral("-="), QStringLiteral("\\*="),
      QStringLiteral("/="), QStringLiteral("\\*")
  };

  const QString braces_patterns[] = {
      QStringLiteral("{"), QStringLiteral("}"), QStringLiteral("\\("), QStringLiteral("\\)")
  };

  const QString numbers_patterns[] = {
      QStringLiteral("(\\d+)\\.(\\d+)")
  };

  keyword_format.setForeground(Qt::darkBlue);
  keyword_format.setFontWeight(QFont::Bold);

  operator_format.setForeground(Qt::darkRed);
  operator_format.setFontWeight(QFont::Bold);

  braces_format.setForeground(Qt::black);
  braces_format.setFontWeight(QFont::Bold);

  numbers_format.setForeground(Qt::darkGreen);
  numbers_format.setFontWeight(QFont::Bold);

  HighlightingRule rule;

  for (auto &keyword: keyword_patterns)
  {
    rule.format = keyword_format;
    rule.pattern = QRegularExpression (keyword);
    highlighting_rules.push_back (rule);
  }

  for (auto &op: operators_patterns)
  {
    rule.format = operator_format;
    rule.pattern = QRegularExpression (op);
    highlighting_rules.push_back (rule);
  }

  for (auto &brace: braces_patterns)
  {
    rule.format = braces_format;
    rule.pattern = QRegularExpression (brace);
    highlighting_rules.push_back (rule);
  }

  for (auto &num: numbers_patterns)
  {
    rule.format = numbers_format;
    rule.pattern = QRegularExpression (num);
    highlighting_rules.push_back (rule);
  }
}

void python_syntax_highlighter::highlightBlock(const QString &text)
{
  for (const HighlightingRule &rule : qAsConst(highlighting_rules)) {
    QRegularExpressionMatchIterator matchIterator = rule.pattern.globalMatch(text);
    while (matchIterator.hasNext()) {
      QRegularExpressionMatch match = matchIterator.next();
      setFormat(match.capturedStart(), match.capturedLength(), rule.format);
    }
  }
  setCurrentBlockState(0);
}