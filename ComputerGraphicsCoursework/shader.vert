#version 330 core

in vec4 vertexColor;
in vec3 vertexPosition;
in vec3 vertexNormal;
in vec2 vertexTexCoord;

out vec4 Position;
out vec3 Normal;
out vec4 Color;
out vec2 texCoord0;

//uniform mat4 ModelViewMatrix;
uniform mat3 NormalMatrix;
uniform mat4 MVP;
uniform mat4 ModelMatrix;


void main()
{
	Position = ModelMatrix * vec4(vertexPosition, 1.0);
	Normal = NormalMatrix * vertexNormal;
	Color = vertexColor;
	texCoord0 = vertexTexCoord;
    gl_Position = MVP * vec4(vertexPosition, 1.0);
}
