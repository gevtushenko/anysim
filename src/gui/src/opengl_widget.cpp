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
  : elements_count (nx * ny)
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
  // cudaGraphicsGLRegisterBuffer (&colors_res, vbo_colors, cudaGraphicsMapFlagsWriteDiscard);

  // d_colors = preprocess_before_colors_fill ();
  // postprocess_after_colors_fill ();

  // auto error = cudaGetLastError ();

  // if (error != cudaSuccess)
  //   std::cout << cudaGetErrorString (error) << std::endl;
#endif

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

    // program->bind();

    // program->setUniformValue ("MVP", mvp);
    // glEnableVertexAttribArray (static_cast<GLuint> (attribute_v_color));
    // glBindBuffer(GL_ARRAY_BUFFER, vbo_colors);
    // glVertexAttribPointer(static_cast<GLuint> (attribute_v_color), 3, GL_FLOAT, GL_FALSE, 0, 0);
    // glEnableVertexAttribArray(static_cast<GLuint> (attribute_coord2d));
    // glBindBuffer(GL_ARRAY_BUFFER, vbo_vertices);
    // glVertexAttribPointer (static_cast<GLuint> (attribute_coord2d), 2, GL_FLOAT, GL_FALSE, 0, 0);
    // glDrawArrays(GL_QUADS, 0, static_cast<int> (elements_count) * 4);
    // glDisableVertexAttribArray(static_cast<GLuint> (attribute_coord2d));
    // glDisableVertexAttribArray(static_cast<GLuint> (attribute_v_color));
    // program->release();

    axes.draw (mvp);

    // tex_program->bind();
    // QString text("123.321e-12");
    // const QChar *qchar = text.data();

    // renderText(qchar, text.size(), 0.0, 0.0, 1.0f, QVector3D(1.0f, 0.0f, 0.0f));
    // tex_program->release();
}