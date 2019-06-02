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
  unsigned int font_size = 18;
  QOpenGLShaderProgram *program = nullptr;
  std::unique_ptr<ft_context> free_type;

  std::map<char, character> characters;

  QOpenGLBuffer tex_vbo;
  QOpenGLVertexArrayObject tex_vao;

  float x_scale = 1.0;
  float y_scale = 1.0;

private:
  const character& get_character (char c);
  float get_char_width (char c);
  float get_char_height (char c);
  void render_character (char c, float &x, float y, float scale);

protected:
  text_renderer ();
  ~text_renderer();

public:
  enum class text_anchor
  {
    left_bottom, bottom_center, right_center
  };

  static text_renderer &instance ();
  void initialize (QObject *parent);
  void finalize ();
  void resize (int width, int height);

  void render_text (
      const std::string &text,
      float x, float y,
      float scale,
      const QMatrix4x4 &mvp,
      text_anchor anchor=text_anchor::left_bottom);
};

#endif //ANYSIM_TEXT_RENDERER_H
