import cx from "@/app/lib/cx";
import { IconArrowBack } from "@tabler/icons-react";
import { IconPlayerStop } from "@tabler/icons-react";
import { IconRefresh } from "@tabler/icons-react";

export default function Form({
    ref,
    input,
    onChange,
    onSubmit,
    isLoading,
    stop,
    error,
    reload,
}: any) {
    return (
        <form
            onSubmit={onSubmit}
            className="relative m-auto flex items-center justify-center gap-4"
            ref={ref}
        >
            <input
                name="prompt"
                type="text"
                value={input}
                placeholder="Escribe cualquier duda aquí"
                required
                className={cx(
                    "h-10 flex-1 rounded-xl pl-4 pr-12 transition md:h-12",
                    "border border-gray-400 text-base text-black",
                    "disabled:bg-gray-100",
                )}
                onChange={onChange}
                disabled={isLoading}
            />

            {!isLoading && !error && (
                <button
                    type="submit"
                    tabIndex={-1}
                    className={cx(
                        "absolute right-3 top-1/2 -translate-y-1/2",
                        "opacity-50",
                    )}
                >
                    <IconArrowBack stroke={1.5} color="black" />
                </button>
            )}

            {isLoading && (
                <button
                    type="button"
                    onClick={() => stop()}
                    tabIndex={-1}
                    className={cx(
                        "absolute right-3 top-1/2 -translate-y-1/2",
                        "opacity-50",
                    )}
                >
                    <IconPlayerStop stroke={1.5} fill="black" />
                </button>
            )}

            {error && (
                <button
                    type="button"
                    onClick={() => reload()}
                    tabIndex={-1}
                    className={cx(
                        "absolute right-3 top-1/2 -translate-y-1/2",
                        "opacity-50",
                    )}
                >
                    <IconRefresh stroke={1.5} color="black" />
                </button>
            )}
        </form>
    );
}
