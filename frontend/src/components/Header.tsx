import Image from "next/image";

export const Header = () => {
  return (
    <header className="bg-white shadow-sm">
      <div className="container mx-auto px-4 py-4 flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <Image
            src="/logo.svg"
            alt="Logo"
            width={40}
            height={40}
            className="h-10 w-auto"
          />
          <h1 className="text-2xl font-bold text-gray-800">
            Plastik Şişe Tespiti
          </h1>
        </div>
      </div>
    </header>
  );
};
