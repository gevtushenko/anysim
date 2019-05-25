//"#version 100\n"  // OpenGL ES 2.0

attribute highp vec4 vertex;
attribute highp vec2 tex_coord;
uniform highp mat4 MVP;
varying highp vec2 tex_coord_0;

void main(void)
{
    gl_Position = MVP * vertex;
    tex_coord_0 = tex_coord;
}
