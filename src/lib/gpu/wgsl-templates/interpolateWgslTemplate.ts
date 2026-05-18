import { WgslStruct } from "./WgslStruct.ts";

export const interpolateWgslTemplate = (strings: TemplateStringsArray, ...interpolations: WgslStruct[]) => {
    const uniqueStructs = new Set<WgslStruct>();
    for (const struct of interpolations) {
        uniqueStructs.add(struct);
    }

    const prelude = uniqueStructs[Symbol.iterator]()
        .map(struct => struct.code)
        .toArray()
        .join("\n\n");

    
    let out = `\
${prelude}

${strings.raw[0]}`;

    for (let i = 0; i < interpolations.length; i++) {
        out += interpolations[i].name;
        out += strings.raw[i + 1];
    }

    return out;
};