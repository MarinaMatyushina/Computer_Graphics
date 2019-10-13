#version 330 core
out vec4 OutColor;

struct Material {
    sampler2D diffuse;
    vec3 specular;    
    float shininess; // Показатель степени зеркального отражения
}; 

struct Light {
    vec4 position;
    vec3 direction;
    float cutOff;
    float outerCutOff;

    vec3 ambient;
    vec3 diffuse;
    vec3 specular;

    float constant;
    float linear;
    float quadratic;
};

in vec4 Position;
in vec3 Normal;
in vec4 Color;
in vec2 texCoord0;

uniform vec3 ViewPosition;
uniform vec3 LightIntensity;
uniform Material material;
uniform Light light;

uniform bool isPhongModel;
uniform bool isLightSource;
uniform bool isFlashLightMode;


void main()
{
    vec3 color;
    if (isPhongModel) {
    	vec3 normal = normalize(Normal);
        vec3 lightDir;
        vec3 view;
        float intensity = 1.0;

        if(light.position.w == 0.0)
        {
            lightDir = normalize(light.position.xyz);
            view = normalize(-Position.xyz);
        }
        else
        {
            lightDir = normalize(light.position.xyz - Position.xyz);
            view = normalize(ViewPosition - Position.xyz);
        }
        vec3 reflect = reflect(-lightDir, normal);

        if (isFlashLightMode) {
            // spotlight (soft edges)
            float theta = dot(lightDir, normalize(-light.direction)); 
            float epsilon = (light.cutOff - light.outerCutOff);
            intensity = clamp((theta - light.outerCutOff) / epsilon, 0.0, 1.0);
        }

        // attenuation
        float distance = length(light.position.xyz - Position.xyz);
        float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * (distance * distance)); 

        // begin with ambient
        color = light.ambient;

        // add diffuse
        float dotNL = max(dot(normal, lightDir), 0.0);
        color += light.diffuse * dotNL * intensity;

        // modulate texture map
        color *= texture2D(material.diffuse, texCoord0).rgb;

        // add specular
        float dotNR = max(dot(view, reflect), 0.0);
        color += pow(dotNR, material.shininess) * light.specular * material.specular * intensity;


        color *= attenuation;
    }
    else if(isLightSource) {
        if (isFlashLightMode) color = vec3(0, 0, 0);
        else color = vec3(1, 1, 1);
    }
    else {
        color = vec3(1, 0, 0);
    }

	OutColor = vec4(LightIntensity * color, 1.0f);
}