import { type WgslItem } from "./WgslItem.ts";

export class WgslConstant<T> implements WgslItem {
    static readonly typeSymbol = Symbol("wgsl constant");

    readonly value: T;
    readonly wgsl: string;

    constructor({
        value,
        wgsl,
    }: {
        value: T,
        wgsl: string,
    }) {
        this.value = value;
        this.wgsl = wgsl;
    }

    get type(): symbol {
        return WgslConstant.typeSymbol;
    }
}