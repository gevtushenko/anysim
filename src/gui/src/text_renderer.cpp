//
// Created by egi on 5/25/19.
//

#include "text_renderer.h"

#include <ft2build.h>
#include FT_FREETYPE_H

#include <QOpenGLTexture>
#include <QFile>

#include <iostream>

class ft_context
{
private:
  FT_Library ft = nullptr;
  FT_Face face = nullptr;

  QByteArray face_content;

public:
  ft_context (unsigned int font_size)
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

  ~character()
  {
    if (texture)
      texture->destroy ();
  }

public:
  std::unique_ptr<QOpenGLTexture> texture;
  QVector2D size;
  QVector2D bearing;
  GLuint advance = 0;
};

text_renderer::text_renderer ()
  : free_type (new ft_context (font_size))
{
  initializeOpenGLFunctions ();

  program.addShaderFromSourceFile (QOpenGLShader::Vertex,   ":/shaders/texture.vert");
  program.addShaderFromSourceFile (QOpenGLShader::Fragment, ":/shaders/texture.frag");
  program.link ();

  program.setUniformValue ("texture", 0);

  tex_vao.create();
  tex_vao.bind();

  tex_vbo.create();
  tex_vbo.bind();
  tex_vbo.allocate(sizeof(GLfloat)*4*4);
  tex_vbo.setUsagePattern(QOpenGLBuffer::DynamicDraw);

  quintptr offset = 0;
  program.setAttributeBuffer("vertex", GL_FLOAT, offset, 2, 4*sizeof(GLfloat));
  program.enableAttributeArray("vertex");

  offset += 2 * sizeof(GLfloat);
  program.setAttributeBuffer("tex_coord", GL_FLOAT, offset, 2, 4*sizeof(GLfloat));
  program.enableAttributeArray("tex_coord");

  tex_vao.release();
}

text_renderer& text_renderer::instance()
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

  GLfloat w = ch.size.x() * scale;
  GLfloat h = ch.size.y() * scale;

  GLfloat xpos = x + ch.bearing.x() * scale;
  GLfloat ypos = y + (ch.size.y() - ch.bearing.y()) * scale + font_size - h;

  GLfloat tex_vertices[] = {
      xpos,     ypos,       0.0, 0.0,
      xpos + w, ypos,       1.0, 0.0,
      xpos,     ypos + h,   0.0, 1.0,
      xpos + w, ypos + h,   1.0, 1.0
  };
  ch.texture->bind();
  tex_vbo.write(0, tex_vertices, sizeof(tex_vertices));

  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
  ch.texture->release ();
  // Now advance cursors for next glyph (note that advance is number of 1/64 pixels)
  // Bitshift by 6 to get value in pixels (2^6 = 64 (divide amount of 1/64th pixels by 64 to get amount of pixels))
  x += (ch.advance >> 6) * scale;
}

void text_renderer::render_text(std::string text, float x, float y, float scale, const QMatrix4x4 &mvp)
{
  program.bind ();
  program.setUniformValue ("MVP", mvp);

  tex_vao.bind ();
  tex_vbo.bind ();

  for (auto c: text)
    render_character (c, x, y, scale);

  tex_vbo.release ();
  tex_vao.release ();
  program.release ();
}