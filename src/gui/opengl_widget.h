//
// Created by egi on 5/11/19.
//

#ifndef FDTD_OPENGL_WIDGET_H
#define FDTD_OPENGL_WIDGET_H

#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>

#include <memory>

class opengl_widget : public QOpenGLWidget, protected QOpenGLFunctions
{
  Q_OBJECT;

public:
  opengl_widget () = delete;
  opengl_widget (unsigned int nx, unsigned int ny);
  ~opengl_widget () override;

protected:
  void initializeGL () override;
  void resizeGL (int width, int height) override;
  void paintGL () override;

private:
  std::unique_ptr<QOpenGLShaderProgram> program;
  GLint attribute_coord2d, attribute_v_color;
  GLuint vbo_vertices, vbo_colors;

  std::unique_ptr<GLfloat[]> colors;
  std::unique_ptr<GLfloat[]> vertices;
  const long int elements_count;
};


#endif //FDTD_OPENGL_WIDGET_H
