//
// Created by egi on 7/5/19.
//

#ifndef ANYSIM_PYTHON_SYNTAX_HIGHLIGHTER_H
#define ANYSIM_PYTHON_SYNTAX_HIGHLIGHTER_H

#include <QRegularExpression>
#include <QSyntaxHighlighter>
#include <QTextCharFormat>

class python_syntax_highlighter : public QSyntaxHighlighter
{
  Q_OBJECT

public:
  explicit python_syntax_highlighter (QTextDocument *parent);

protected:
  void highlightBlock(const QString &text) override;

private:
  struct HighlightingRule
  {
    QRegularExpression pattern;
    QTextCharFormat format;
  };

  QVector<HighlightingRule> highlighting_rules;
  QTextCharFormat keyword_format, operator_format, braces_format, numbers_format;
};

#endif //ANYSIM_PYTHON_SYNTAX_HIGHLIGHTER_H
