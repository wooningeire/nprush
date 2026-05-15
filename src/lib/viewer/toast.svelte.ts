import { SvelteSet } from "svelte/reactivity";

export type ToastKind = "info" | "warning" | "error" | "success";

export type Toast = {
    message: string,
    kind: ToastKind,
};

export const toasts = $state(new SvelteSet<Toast>());

export const showToast = (message: string, kind: ToastKind = "info", duration = 4000) => {
    const toast = {
        message,
        kind
    };

    toasts.add(toast);

    if (duration > 0) {
        dismissToast(toast, duration);
    }

    return toast;
};

export const dismissToast = (toast: Toast, delay = 0) => {
    if (delay === 0) {
        toasts.delete(toast);
    } else {
        setTimeout(() => {
            toasts.delete(toast);
        }, delay);
    }
};
