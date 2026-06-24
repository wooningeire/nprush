import { WgslStruct } from "../wgsl-templates/WgslStruct.ts";

export const Splat = WgslStruct.fromCode(/* wgsl */`\
struct Splat {
    pos_sx: vec4f,
    color: vec4f,
    quat: vec4f,
    sy_shape: vec4f,
    sh1_r: vec4f,
    sh1_g: vec4f,
    sh1_b: vec4f,
    sh1_a: vec4f,
}`);