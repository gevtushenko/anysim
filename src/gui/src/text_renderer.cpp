//
// Created by egi on 5/25/19.
//

#include "text_renderer.h"
#include "cpp/common_funcs.h"

#include <ft2build.h>
#include FT_FREETYPE_H

#include <QDesktopWidget>
#include <QOpenGLTexture>
#include <QApplication>
#include <QScreen>
#include <QWidget>
#include <QFile>

#include <iostream>
#include <cmath>

class ft_context
{
private:
  FT_Library ft = nullptr;
  FT_Face face = nullptr;

  QByteArray face_content;

public:
  explicit ft_context (unsigned int font_size)
  {
    QFile font_face (":/fonts/opensans/OpenSans-Regular.ttf");
    font_face.open(QIODevice::ReadOnly);
    face_content = font_face.readAll();


    if(FT_Init_FreeType(&ft)) {
      std::cerr << "Could not init freetype library\n";
    }

    if(FT_New_Memory_Face(ft, reinterpret_cast<const FT_Byte*> (face_content.data ()), face_content.count (), 0, &face)) {
      std::cerr << "Could not open font\n";
    }

    FT_Set_Pixel_Sizes(face, 0, font_size);
  }

  bool load_glyph (char c)
  {
    return FT_Load_Char (face, c, FT_LOAD_RENDER);
  }

  FT_GlyphSlot glyph ()
  {
    return face->glyph;
  }

  short units_per_em ()
  {
    return face->units_per_EM;
  }

  signed long y_min ()
  {
    return face->bbox.yMin;
  }

  signed long y_max ()
  {
    return face->bbox.yMax;
  }
};

class character
{
public:
  character () = default;
  character (character &&rhs) = default;

  explicit character (FT_GlyphSlot glyph)
    : texture (new QOpenGLTexture (QOpenGLTexture::Target2D))
  {
    size = QVector2D(glyph->bitmap.width, glyph->bitmap.rows);
    bearing = QVector2D(glyph->bitmap_left, glyph->bitmap_top),
    advance = glyph->advance.x;

    texture->setFormat(QOpenGLTexture::R8_UNorm);
    texture->setSize(glyph->bitmap.width, glyph->bitmap.rows);
    texture->allocateStorage();
    texture->setData(QOpenGLTexture::Red, QOpenGLTexture::UInt8, glyph->bitmap.buffer);
    texture->setMagnificationFilter(QOpenGLTexture::Linear);
    texture->setMinificationFilter(QOpenGLTexture::Linear);
    texture->setWrapMode(QOpenGLTexture::ClampToEdge);
  }

public:
  std::unique_ptr<QOpenGLTexture> texture;
  QVector2D size;
  QVector2D bearing;
  GLuint advance = 0;
};

text_renderer::text_renderer ()
  : free_type (new ft_context (font_size))
  , tex_vbo (QOpenGLBuffer::VertexBuffer)
{
}

text_renderer::~text_renderer() = default;

void text_renderer::initialize (QObject *parent)
{
  initializeOpenGLFunctions ();

  cpp_unreferenced (parent);
  program = new QOpenGLShaderProgram (parent);
  program->addShaderFromSourceFile (QOpenGLShader::Vertex,   ":/shaders/texture.vert");
  program->addShaderFromSourceFile (QOpenGLShader::Fragment, ":/shaders/texture.frag");
  if (!program->link ())
    std::cout << "Can't link" << std::endl;

  program->setUniformValue ("qt_Texture0", 0);

  tex_vao.create();
  tex_vao.bind();

  tex_vbo.create();
  tex_vbo.bind();
  tex_vbo.allocate(sizeof(GLfloat)*4*4);
  tex_vbo.setUsagePattern(QOpenGLBuffer::DynamicDraw);

  quintptr offset = 0;
  program->setAttributeBuffer("qt_Vertex", GL_FLOAT, offset, 2, 4*sizeof(GLfloat));
  program->enableAttributeArray("qt_Vertex");

  offset += 2 * sizeof(GLfloat);
  program->setAttributeBuffer("qt_TexCoord", GL_FLOAT, offset, 2, 4*sizeof(GLfloat));
  program->enableAttributeArray("qt_TexCoord");

  tex_vao.release();
}

void text_renderer::finalize ()
{
  characters.clear ();
}

void text_renderer::resize(int width, int height)
{
  x_scale = 2.0f / width;
  y_scale = 2.0f / height;
}

text_renderer& text_renderer::instance ()
{
  static text_renderer tr;
  return tr;
}

const character& text_renderer::get_character(char c)
{
  auto it = characters.find (c);
  if (it == characters.end ())
  {
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    if (free_type->load_glyph (c)) {
      std::cerr << "Free type failed to load glyph.";
      return get_character ('?');
    }

    characters.emplace (c, character (free_type->glyph ()));
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    return characters[c];
  }

  return it->second;
}

void text_renderer::render_character(char c, float &x, float y, float scale)
{
  auto &ch = get_character (c);
  if (!ch.texture)
  {
    std::cout << "Error" << std::endl;
    return;
  }

  GLfloat w = ch.size.x() * x_scale * scale;
  GLfloat h = ch.size.y() * y_scale * scale;

  GLfloat xpos = x + ch.bearing.x() * x_scale * scale;
  GLfloat ypos = y + h - (h - ch.bearing.y () * y_scale * scale);

  GLfloat tex_vertices[] = {
      xpos,     ypos    ,   0.0, 0.0,
      xpos + w, ypos    ,   1.0, 0.0,
      xpos,     ypos - h,   0.0, 1.0,
      xpos + w, ypos - h,   1.0, 1.0
  };
  ch.texture->bind();
  tex_vbo.write(0, tex_vertices, sizeof(tex_vertices));
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
  ch.texture->release ();
  // Now advance cursors for next glyph (note that advance is number of 1/64 pixels)
  // Bitshift by 6 to get value in pixels (2^6 = 64 (divide amount of 1/64th pixels by 64 to get amount of pixels))
  x += (ch.advance >> 6u) * x_scale * scale;
}

float text_renderer::get_char_width (char c)
{
  auto &ch = get_character (c);
  if (!ch.texture)
    return 0.0f;

  return ch.size.x () * x_scale;
}

float text_renderer::get_char_height (char c)
{
  auto &ch = get_character (c);
  if (!ch.texture)
    return 0.0f;

  return ch.size.y () * y_scale;
}

void text_renderer::render_text (
    const std::string &text,
    float x, float y,
    float scale,
    const QMatrix4x4 &mvp,
    text_anchor anchor)
{
  float y_offset = 0.0;

  if (anchor == text_anchor::bottom_center)
  {
    float x_offset = 0.0;

    for (auto c: text)
    {
      float height = get_char_height (c);
      if (height > y_offset)
        y_offset = height;

      x_offset += get_char_width (c);
    }

    x -= x_offset / 2.0;
  }
  else if (anchor == text_anchor::right_center)
  {
    float x_offset = 0.0;

    for (auto c: text)
    {
      float height = get_char_height (c);
      if (height > y_offset)
        y_offset = height;

      x_offset += get_char_width (c);
    }

    y_offset /= 2.0;
    x -= x_offset;
  }

  program->bind ();
  program->setUniformValue ("qt_ModelViewProjectionMatrix", mvp);

  QVector3D color (0.0, 0.0, 0.0);
  program->setUniformValue ("textColor", color);

  tex_vao.bind ();
  tex_vbo.bind ();

  for (auto c: text)
    render_character (c, x, y - y_offset, scale);

  tex_vbo.release ();
  tex_vao.release ();
  program->release ();
}