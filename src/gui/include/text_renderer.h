//
// Created by egi on 5/25/19.
//

#ifndef ANYSIM_TEXT_RENDERER_H
#define ANYSIM_TEXT_RENDERER_H

#include <QOpenGLVertexArrayObject>
#include <QOpenGLShaderProgram>
#include <QOpenGLFunctions>
#include <QOpenGLBuffer>

#include <memory>
#include <map>

class character;
class ft_context;

class text_renderer : protected QOpenGLFunctions
{
private:
  unsigned int font_size = 48;
  std::map<char, character> characters;

  QOpenGLBuffer tex_vbo;
  QOpenGLShaderProgram program;
  QOpenGLVertexArrayObject tex_vao;

  std::unique_ptr<ft_context> free_type;

private:
  const character& get_character (char c);
  void render_character (char c, float &x, float y, float scale);

protected:
  text_renderer ();

public:
  static text_renderer &instance ();
  void render_text (std::string text, float x, float y, float scale, const QMatrix4x4 &mvp);
};

#endif //ANYSIM_TEXT_RENDERER_H
