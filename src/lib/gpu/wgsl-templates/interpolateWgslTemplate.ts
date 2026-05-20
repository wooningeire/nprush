import { WgslConstant } from "./WgslConstant.ts";
import type { WgslItem } from "./WgslItem.ts";
import { WgslStruct } from "./WgslStruct.ts";

export const interpolateWgslTemplate = (strings: TemplateStringsArray, ...interpolations: WgslItem[]) => {
    const uniqueStructs = new Set<WgslStruct>();
    const uniqueConstants = new Set<WgslConstant<unknown>>();
    
    for (const item of interpolations) {
        switch (item.type) {
            case WgslStruct.typeSymbol:
                uniqueStructs.add(item as WgslStruct);
                break;
                
            case WgslConstant.typeSymbol:
                uniqueConstants.add(item as WgslConstant<unknown>);
                break;

            default:
                throw new Error(`Unknown WgslItem type: ${item.type.description}`);
        }
    }

    const prelude = uniqueStructs[Symbol.iterator]()
        .map(struct => struct.code)
        .toArray()
        .join("\n\n");

    
    let out = `\
${prelude}

${strings.raw[0]}`;

    for (let i = 0; i < interpolations.length; i++) {
        switch (interpolations[i].type) {
            case WgslStruct.typeSymbol:
                out += (interpolations[i] as WgslStruct).name;
                break;

            case WgslConstant.typeSymbol:
                out += (interpolations[i] as WgslConstant<unknown>).wgsl;
                break;

            default:
                throw new Error(`Unknown WgslItem type: ${interpolations[i].type.description}`);
        }

        out += strings.raw[i + 1];
    }

    return out;
};