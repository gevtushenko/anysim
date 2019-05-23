//
// Created by egi on 5/11/19.
//

#ifndef FDTD_OPENGL_WIDGET_H
#define FDTD_OPENGL_WIDGET_H

#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>

#include <QOpenGLBuffer>
#include <QOpenGLTexture>
#include <QOpenGLVertexArrayObject>

#include <ft2build.h>
#include FT_FREETYPE_H

#include <memory>
#include <functional>

#ifdef GPU_BUILD
#include <cuda_gl_interop.h>
#endif

class Character
{
public:
    Character() {}

    Character(QOpenGLTexture *texture, QVector2D size, QVector2D bearing, GLuint advance) {
        this->texture = texture;
        this->size = size;
        this->bearing = bearing;
        this->advance = advance;
    }

    QOpenGLTexture      *texture;
    QVector2D   size;
    QVector2D   bearing;
    GLuint      advance;
};

class opengl_widget : public QOpenGLWidget, protected QOpenGLFunctions
{
  Q_OBJECT;

public:
  opengl_widget () = delete;
  opengl_widget(unsigned int nx,
                unsigned int ny,
                float x_size,
                float y_size);
  ~opengl_widget () override;

  float *get_colors (bool use_gpu);

  float * preprocess_before_colors_fill ();
  void postprocess_after_colors_fill ();

public slots:
  void update_colors (bool use_gpu);

protected:
  void initializeGL () override;
  void resizeGL (int width, int height) override;
  void paintGL () override;
  void wheelEvent (QWheelEvent *event) override;

private:
    Character getCharacter(QChar character);
    void renderText(const QChar *text, int length, GLfloat x, GLfloat y, GLfloat scale, QVector3D color);

private:
  std::unique_ptr<QOpenGLShaderProgram> program;
  std::unique_ptr<QOpenGLShaderProgram> tex_program;

  QByteArray face_content;

  GLint attribute_coord2d, attribute_v_color;
  GLuint vbo_vertices, vbo_colors;

  QOpenGLBuffer tex_vbo;
  QOpenGLVertexArrayObject tex_vao;

  const long int elements_count;
  const int vertices_per_element = 4;
  const int coords_per_vertex = 2;
  const int vertex_data_per_element = vertices_per_element * coords_per_vertex;
  const int colors_per_vertex = 3;
  const int color_data_per_element = colors_per_vertex * vertices_per_element;

#ifdef GPU_BUILD
  cudaGraphicsResource_t colors_res;
#endif

  std::unique_ptr<GLfloat[]> colors;
  std::unique_ptr<GLfloat[]> vertices;

  FT_Library ft;
  FT_Face face;

  float *d_colors = nullptr;

  QMatrix4x4 mvp;
};

#endif //FDTD_OPENGL_WIDGET_H
