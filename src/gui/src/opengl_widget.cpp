//
// Created by egi on 5/11/19.
//

#include "gui/include/opengl_widget.h"

#include <QDirIterator>
#include <QWheelEvent>

#include <iostream>
#include <cmath>

const char *tex_vs_source =
        "attribute highp vec4 qt_Vertex;\n"
        "attribute highp vec2 qt_TexCoord;\n"
        "uniform highp mat4 qt_ModelViewProjectionMatrix;\n"
        "varying highp vec2 qt_TexCoord0;\n"
        "void main(void)\n"
        "{\n"
        "    gl_Position = qt_ModelViewProjectionMatrix * qt_Vertex;\n"
        "    qt_TexCoord0 = qt_TexCoord;\n"
        "}";

const char *tex_fs_source =
        "uniform sampler2D qt_Texture0;\n"
        "uniform vec3 textColor;\n"
        "varying highp vec2 qt_TexCoord0;\n"
        "void main(void)\n"
        "{\n"
        "    vec4 sampled = vec4(1.0, 1.0, 1.0, texture2D(qt_Texture0, qt_TexCoord0).r);\n"
        "    gl_FragColor = vec4(textColor, 1.0) * sampled;\n"
        "}";

opengl_widget::opengl_widget (unsigned int nx, unsigned int ny, float x_size, float y_size)
  : tex_vbo (QOpenGLBuffer::VertexBuffer)
  , elements_count (nx * ny)
  , colors (new GLfloat[color_data_per_element * elements_count])
  , vertices (new GLfloat[vertex_data_per_element * elements_count])
{
  GLfloat max_width  = (r_x - l_x) * (x_size >= y_size ? 1.0f : x_size / y_size);
  GLfloat max_height = (t_y - b_y) * (y_size >  x_size ? 1.0f : y_size / x_size);

  GLfloat dx = max_width / static_cast<GLfloat> (nx);
  GLfloat dy = max_height / static_cast<GLfloat> (ny);

  static GLfloat _colors[] =
      {
          1.0, 1.0, 0.0,
          0.0, 0.0, 1.0,
          1.0, 0.0, 0.0,
          1.0, 0.0, 1.0,
      };

  for (unsigned int j = 0; j < ny; j++)
  {
    for (unsigned int i = 0; i < nx; i++)
    {
      const unsigned int vert_offset = static_cast<unsigned int> (vertex_data_per_element) * (j * nx + i);

      vertices[vert_offset + 0] = l_x + dx * (i + 0); vertices[vert_offset + 1] = b_y + dy * (j + 1);
      vertices[vert_offset + 2] = l_x + dx * (i + 0); vertices[vert_offset + 3] = b_y + dy * (j + 0);
      vertices[vert_offset + 4] = l_x + dx * (i + 1); vertices[vert_offset + 5] = b_y + dy * (j + 0);
      vertices[vert_offset + 6] = l_x + dx * (i + 1); vertices[vert_offset + 7] = b_y + dy * (j + 1);

      const unsigned int color_offset = static_cast<unsigned int> (color_data_per_element) * (j * nx + i);
      std::copy_n (_colors, color_data_per_element, colors.get () + color_offset);
    }
  }

  QFile font_face (":/fonts/opensans/OpenSans-Regular.ttf");
  font_face.open(QIODevice::ReadOnly);
  face_content = font_face.readAll();


  if(FT_Init_FreeType(&ft)) {
    std::cerr << "Could not init freetype library\n";
  }

  if(FT_New_Memory_Face(ft, reinterpret_cast<const FT_Byte*> (face_content.data ()), face_content.count (), 0, &face)) {
    std::cerr << "Could not open font\n";
  }

  FT_Set_Pixel_Sizes(face, 0, 48);

  std::cout << max_width << " - " << max_height << std::endl;
}

opengl_widget::~opengl_widget ()
{
#ifdef GPU_BUILD
  cudaGraphicsUnregisterResource (colors_res);
#endif
  glDeleteBuffers (1, &vbo_vertices);
  glDeleteBuffers (1, &vbo_colors);
}

// GLfloat *opengl_widget::get_colors ()
// {
//   return colors.get ();
// }

float* opengl_widget::preprocess_before_colors_fill()
{
#ifdef GPU_BUILD
  size_t size = 0;
  float *colors_ptr = nullptr;
  cudaGraphicsMapResources (1, &colors_res);
  cudaGraphicsResourceGetMappedPointer ((void**) &colors_ptr, &size, colors_res);

  auto error = cudaGetLastError ();

  if (error != cudaSuccess)
    std::cout << cudaGetErrorString (error) << std::endl;

  return colors_ptr;
#else
  return nullptr;
#endif
}

void opengl_widget::postprocess_after_colors_fill()
{
#ifdef GPU_BUILD
  cudaGraphicsUnmapResources (1, &colors_res);
#endif
}

void opengl_widget::initializeGL()
{
  initializeOpenGLFunctions ();
  program = std::make_unique<QOpenGLShaderProgram> (this);
  program->addShaderFromSourceFile (QOpenGLShader::Vertex,   ":/shaders/map_2d.vert");
  program->addShaderFromSourceFile (QOpenGLShader::Fragment, ":/shaders/map_2d.frag");
  program->link ();
  attribute_coord2d = program->attributeLocation ("coord2d");
  attribute_v_color = program->attributeLocation ("v_color");

  /// VBO Handling
  const int glfloat_size = sizeof (GLfloat);
  const long int vertices_array_size = elements_count * vertex_data_per_element * glfloat_size;
  glGenBuffers (1, &vbo_vertices);
  glBindBuffer (GL_ARRAY_BUFFER, vbo_vertices);
  glBufferData (GL_ARRAY_BUFFER, vertices_array_size, vertices.get (), GL_DYNAMIC_DRAW);

  const long int colors_array_size = elements_count * color_data_per_element * glfloat_size;
  glGenBuffers (1, &vbo_colors);
  glBindBuffer (GL_ARRAY_BUFFER, vbo_colors);
  glBufferData (GL_ARRAY_BUFFER, colors_array_size, colors.get (), GL_DYNAMIC_DRAW);

#ifdef GPU_BUILD
  cudaGraphicsGLRegisterBuffer (&colors_res, vbo_colors, cudaGraphicsMapFlagsWriteDiscard);

  d_colors = preprocess_before_colors_fill ();
  postprocess_after_colors_fill ();

  auto error = cudaGetLastError ();

  if (error != cudaSuccess)
    std::cout << cudaGetErrorString (error) << std::endl;
#endif

    tex_program = std::make_unique<QOpenGLShaderProgram> (this);
    tex_program->addShaderFromSourceCode (QOpenGLShader::Vertex, tex_vs_source);
    tex_program->addShaderFromSourceCode (QOpenGLShader::Fragment, tex_fs_source);
    tex_program->link ();

    tex_program->setUniformValue("qt_Texture0", 0);

    tex_vao.create();
    tex_vao.bind();

    tex_vbo.create();
    tex_vbo.bind();
    tex_vbo.allocate(sizeof(GLfloat)*4*4);
    tex_vbo.setUsagePattern(QOpenGLBuffer::DynamicDraw);

    quintptr offset = 0;
    tex_program->setAttributeBuffer("qt_Vertex", GL_FLOAT, offset, 2, 4*sizeof(GLfloat));
    tex_program->enableAttributeArray("qt_Vertex");
    offset += 2 * sizeof(GLfloat);
    tex_program->setAttributeBuffer("qt_TexCoord", GL_FLOAT, offset, 2, 4*sizeof(GLfloat));
    tex_program->enableAttributeArray("qt_TexCoord");

    tex_vao.release();

  mvp.setToIdentity ();

  axes.init (44, 44, l_x, r_x, b_y, t_y);
}

float *opengl_widget::get_colors (bool use_gpu)
{
  return use_gpu ? d_colors : colors.get ();
}

void opengl_widget::resizeGL(int width, int height)
{
    (void) width;
    (void) height;
}

void opengl_widget::update_colors (bool use_gpu)
{
  if (!use_gpu)
  {
    const int glfloat_size = sizeof (GLfloat);
    const long int colors_array_size = elements_count * color_data_per_element * glfloat_size;

    glBindBuffer (GL_ARRAY_BUFFER, vbo_colors);
    glBufferData (GL_ARRAY_BUFFER, colors_array_size, colors.get (), GL_DYNAMIC_DRAW);
  }

  update ();
}

Character opengl_widget::getCharacter(QChar character)
{
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    if (FT_Load_Char(face, character.unicode(), FT_LOAD_RENDER)) {
        qDebug() << "ERROR::FREETYTPE: Failed to load Glyph";
        return Character();
    }
    QOpenGLTexture *texture = new QOpenGLTexture(QOpenGLTexture::Target2D);
    texture->setFormat(QOpenGLTexture::R8_UNorm);
    texture->setSize(face->glyph->bitmap.width, face->glyph->bitmap.rows);
    texture->allocateStorage();
    texture->setData(QOpenGLTexture::Red, QOpenGLTexture::UInt8,
                     face->glyph->bitmap.buffer);
    texture->setMagnificationFilter(QOpenGLTexture::Linear);
    texture->setMinificationFilter(QOpenGLTexture::Linear);
    texture->setWrapMode(QOpenGLTexture::ClampToEdge);
    Character newChar(
            texture,
            QVector2D(face->glyph->bitmap.width, face->glyph->bitmap.rows),
            QVector2D(face->glyph->bitmap_left, face->glyph->bitmap_top),
            face->glyph->advance.x
    );
    // m_typeList.insert(character, newChar);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    return newChar;
}

void opengl_widget::renderText(const QChar *text, int length, GLfloat x, GLfloat y, GLfloat scale, QVector3D color)
{
    int fontsize = 48;

    QMatrix4x4 matrix;
    matrix.ortho(QRect(0, 0, 800, 600));
    tex_program->setUniformValue("qt_ModelViewProjectionMatrix", matrix);
    tex_program->setUniformValue("textColor", color);
    tex_vao.bind();
    for (int i=0; i<length; i++) {
        Character ch = getCharacter(text[i]);
        GLfloat w = ch.size.x() * scale;
        GLfloat h = ch.size.y() * scale;

        GLfloat xpos = x + ch.bearing.x() * scale;
        GLfloat ypos = y + (ch.size.y() - ch.bearing.y()) * scale + fontsize - h;

        GLfloat tex_vertices[] = {
                xpos,     ypos,       0.0, 0.0,
                xpos + w, ypos,       1.0, 0.0,
                xpos,     ypos + h,   0.0, 1.0,
                xpos + w, ypos + h,   1.0, 1.0
        };
        ch.texture->bind();
        tex_vbo.write(0, tex_vertices, sizeof(tex_vertices));

        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        // Now advance cursors for next glyph (note that advance is number of 1/64 pixels)
        // Bitshift by 6 to get value in pixels (2^6 = 64 (divide amount of 1/64th pixels by 64 to get amount of pixels))
        x += (ch.advance >> 6) * scale;
    }
    tex_vao.release();
    matrix.setToIdentity();
    tex_program->setUniformValue("qt_ModelViewProjectionMatrix", matrix);
}

void opengl_widget::wheelEvent(QWheelEvent *event)
{
  if (event->delta () != 0)
  {
    mvp.scale (std::pow (0.9, -static_cast<float> (event->delta ()) / 120));
    update ();
  }
}

void opengl_widget::paintGL()
{
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glClear(GL_COLOR_BUFFER_BIT);
    glClearColor (1.0f, 1.0f, 1.0f, 1.0f);

    program->bind();

    program->setUniformValue ("MVP", mvp);
    glEnableVertexAttribArray (static_cast<GLuint> (attribute_v_color));
    glBindBuffer(GL_ARRAY_BUFFER, vbo_colors);
    glVertexAttribPointer(static_cast<GLuint> (attribute_v_color), 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(static_cast<GLuint> (attribute_coord2d));
    glBindBuffer(GL_ARRAY_BUFFER, vbo_vertices);
    glVertexAttribPointer (static_cast<GLuint> (attribute_coord2d), 2, GL_FLOAT, GL_FALSE, 0, 0);
    glDrawArrays(GL_QUADS, 0, static_cast<int> (elements_count) * 4);
    glDisableVertexAttribArray(static_cast<GLuint> (attribute_coord2d));
    glDisableVertexAttribArray(static_cast<GLuint> (attribute_v_color));
    program->release();

    axes.draw (mvp);

    // tex_program->bind();
    // QString text("123.321e-12");
    // const QChar *qchar = text.data();

    // renderText(qchar, text.size(), 0.0, 0.0, 1.0f, QVector3D(1.0f, 0.0f, 0.0f));
    // tex_program->release();
}