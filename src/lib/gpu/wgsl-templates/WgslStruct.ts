import type { WgslItem } from "./WgslItem.ts";

export class WgslStruct implements WgslItem {
    static readonly typeSymbol = Symbol("wgsl struct");

    readonly name: string;
    readonly code: string;

    private constructor({
        name,
        code,
    }: {
        name: string,
        code: string,
    }) {
        this.name = name;
        this.code = code;
    }

    static fromCode(code: string) {
        const nameResult = code.match(/struct\s+(.+)\s+\{/m);
        if (nameResult === null) throw new Error("could not parse wgsl struct");

        return new WgslStruct({
            name: nameResult[1],
            code,
        });
    }

    get type(): symbol {
        return WgslStruct.typeSymbol;
    }
}
